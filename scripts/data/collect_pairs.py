"""E17 ([26]): collect the paired junction routes with PDM-Lite (carla_garage, tia's fork, branch
leaderboard_2) on the cluster, in the folder layout our training code reads. Training towns only;
routes from scripts/data/pair_junctions.py, checked by scripts/data/check_pair_routes.py (collect.json).

run     one CARLA server per slot (--gpu-ids 0,0,1,1 = 2 servers on GPU 0, 2 on GPU 1), each slot takes
        routes from a shared queue and runs carla_garage's leaderboard_evaluator_local.py with
        team_code/data_agent.py on one route XML at a time (the server stays up; it is restarted when it
        dies). carla_garage switches (env, default off there, on here):
          DATAGEN=1             PDM-Lite data collection
          RGB_ONLY=1            only rgb/, rgb_augmented/, measurements/, results.json.gz
          KEEP_ROUTE_WEATHER=1  weather from the route XML (same for all routes of a group)
          NO_BACKGROUND=1       no BackgroundActivity traffic (paired frames before the junction look alike)
          ALL_GREEN=1           all traffic lights green and frozen (as in CARLA-F)
        traffic-manager seed = one per group, so both routes of a pair get the same seeded randomness
        (camera augmentation of rgb_augmented, parked cars).
        Output (--out, default database/e17_pairs):
          data/simlingo/e17/routes_training/{paired,unpaired}/<Town>_Rep0_<route>_route<id>_<date>/
              rgb/NNNN.jpg  rgb_augmented/NNNN.jpg  measurements/NNNN.json.gz  results.json.gz  (+ PDM-Lite logs)
          results/{paired,unpaired}/<route>.json   leaderboard checkpoint (resume / done marker)
          logs/<route>.{out,err}  logs/carla_<slot>.log  collect_status.jsonl (one line per attempt)
        Resubmitting the same command resumes: finished routes are skipped; folders of unfinished attempts
        are moved to <out>/failed/ before a retry.
report  per route: frames, score, clean as SimLingoBase._route_is_clean, command sequence; per group:
        start distance between the routes, frames before the junction.

    python scripts/data/collect_pairs.py run --garage ../carla_garage --groups Town12_j20148_r913_l-1 --gpu-ids 0,0
    python scripts/data/collect_pairs.py run --garage ../carla_garage --gpu-ids 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3
    python scripts/data/collect_pairs.py report --out database/e17_pairs
"""
import argparse, glob, gzip, json, os, queue, shutil, signal, socket, subprocess, sys, threading, time, zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KINDS = ("paired", "unpaired")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def route_jobs(pairs_dir, groups=None, limit=0):
    """[(kind, group, route name, xml path, seed)] in collect.json order, both routes of a group adjacent."""
    collect = json.load(open(pairs_dir / "check" / "collect.json"))
    jobs = []
    for kind in KINDS:
        for g, names in collect[kind].items():
            if groups and g not in groups:
                continue
            seed = zlib.crc32(g.encode()) % 1000000
            for n in names:
                town = n.split("_", 1)[0]
                jobs.append((kind, g, n, pairs_dir / "routes" / town / f"{n}.xml", seed))
    if groups:
        missing = set(groups) - {j[1] for j in jobs}
        if missing:
            sys.exit(f"groups not in collect.json: {sorted(missing)}")
    if limit:
        keep = list(dict.fromkeys(j[1] for j in jobs))[:limit]
        jobs = [j for j in jobs if j[1] in keep]
    return jobs


def route_dirs(out, kind, name):
    return sorted(glob.glob(str(out / "data/simlingo/e17/routes_training" / kind / f"*_Rep0_{name}_route*")))


def finished(out, kind, name):
    """Done = leaderboard checkpoint says 1/1 routes and exactly one route folder with results.json.gz."""
    ck = out / "results" / kind / f"{name}.json"
    try:
        prog = json.load(open(ck))["_checkpoint"]["progress"]
    except Exception:
        return False
    dirs = [d for d in route_dirs(out, kind, name) if os.path.isfile(f"{d}/results.json.gz")]
    return len(prog) == 2 and prog[0] >= prog[1] and len(dirs) == 1


def port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


class Slot:
    def __init__(self, idx, gpu, port, args):
        self.idx, self.gpu, self.port, self.args = idx, gpu, port, args
        self.tm_port = port + 50  # CARLA itself uses port, port+1, port+2
        self.server = None

    def start_carla(self):
        self.stop_carla()
        carla = Path(os.environ["CARLA_ROOT"]) / "CarlaUE4.sh"
        cmd = [str(carla), "-RenderOffScreen", "-nosound", f"-carla-rpc-port={self.port}",
               f"-graphicsadapter={self.gpu}"]
        lf = open(self.args.out / "logs" / f"carla_{self.idx}.log", "a")
        self.server = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        log(f"slot {self.idx}: CARLA gpu={self.gpu} port={self.port} pid={self.server.pid}")
        t0 = time.time()
        while time.time() - t0 < self.args.carla_wait:
            if self.server.poll() is not None:
                return False
            if port_open(self.port) and time.time() - t0 > 20:
                time.sleep(10)
                return True
            time.sleep(2)
        return port_open(self.port)

    def stop_carla(self):
        if self.server is not None:
            try:
                os.killpg(self.server.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.server.wait()
            self.server = None
        # anything still bound to this slot's rpc port (CarlaUE4-Linux-Shipping started by the .sh)
        subprocess.run(f"pkill -9 -f -- '-carla-rpc-port={self.port}( |$)'", shell=True)
        time.sleep(3)

    def alive(self):
        return self.server is not None and self.server.poll() is None and port_open(self.port)

    def run_route(self, kind, group, name, xml, seed):
        a, out, garage = self.args, self.args.out, self.args.garage
        for d in route_dirs(out, kind, name):  # leftovers of an unfinished attempt
            dst = out / "failed" / f"{Path(d).name}_{int(time.time())}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(d, dst)
        ck = out / "results" / kind / f"{name}.json"
        ck.parent.mkdir(parents=True, exist_ok=True)
        if ck.exists():
            ck.unlink()
        save = out / "data/simlingo/e17/routes_training" / kind
        save.mkdir(parents=True, exist_ok=True)
        carla_root = os.environ["CARLA_ROOT"]
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": os.pathsep.join([f"{carla_root}/PythonAPI/carla", str(garage / "leaderboard_autopilot"),
                                           str(garage / "scenario_runner_autopilot"), str(garage / "team_code")]),
            "SCENARIO_RUNNER_ROOT": str(garage / "scenario_runner_autopilot"),
            "LEADERBOARD_ROOT": str(garage / "leaderboard_autopilot"),
            "CHALLENGE_TRACK_CODENAME": "MAP", "DEBUG_CHALLENGE": "0", "DATAGEN": "1",
            "SAVE_PATH": str(save), "TOWN": name.split("_", 1)[0], "REPETITION": "0",
            "RGB_ONLY": "1", "KEEP_ROUTE_WEATHER": "1", "NO_BACKGROUND": "1", "ALL_GREEN": str(int(a.all_green)),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
        })
        cmd = [sys.executable, "-u", str(garage / "leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py"),
               f"--port={self.port}", f"--traffic-manager-port={self.tm_port}", f"--traffic-manager-seed={seed}",
               f"--routes={xml}", "--repetitions=1", "--track=MAP", f"--checkpoint={ck}",
               f"--agent={garage / 'team_code/data_agent.py'}", f"--agent-config={xml}", "--debug=0", "--resume=0",
               f"--timeout={a.client_timeout}"]
        t0 = time.time()
        with open(out / "logs" / f"{name}.out", "a") as fo, open(out / "logs" / f"{name}.err", "a") as fe:
            p = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env, cwd=garage, start_new_session=True)
            try:
                rc = p.wait(timeout=a.route_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                rc = "timeout"
        ok = finished(out, kind, name)
        with open(out / "collect_status.jsonl", "a") as f:
            f.write(json.dumps(dict(route=name, group=group, kind=kind, slot=self.idx, rc=rc, ok=ok,
                                    sec=round(time.time() - t0), t=time.strftime("%m-%d %H:%M:%S"))) + "\n")
        log(f"slot {self.idx}: {'DONE' if ok else 'FAIL'} {name} rc={rc} {time.time() - t0:.0f}s")
        return ok


def cmd_run(a):
    a.out = (REPO / a.out).resolve() if not os.path.isabs(a.out) else Path(a.out)
    a.garage = Path(a.garage).resolve()
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    if not (a.garage / "team_code/data_agent.py").exists():
        sys.exit(f"no carla_garage at {a.garage}")
    if "RGB_ONLY" not in open(a.garage / "team_code/data_agent.py").read():
        sys.exit("carla_garage without the E17 switches (RGB_ONLY ...): git pull the fork")
    jobs = route_jobs(REPO / a.pairs, set(a.groups.split(",")) if a.groups else None, a.limit_groups)
    todo = [j for j in jobs if not finished(a.out, j[0], j[2])]
    log(f"{len(jobs)} routes ({len({j[1] for j in jobs})} groups), {len(jobs) - len(todo)} already done, {len(todo)} to do")
    q = queue.Queue()
    for j in todo:
        q.put((j, 0))
    gpus = [int(g) for g in a.gpu_ids.split(",")]
    slots = [Slot(i, g, a.base_port + 100 * i, a) for i, g in enumerate(gpus)]
    stats = {"done": 0, "fail": 0}
    lock = threading.Lock()

    def worker(slot):
        fails_in_row = 0
        while True:
            try:
                job, tries = q.get_nowait()
            except queue.Empty:
                break
            if not slot.alive():
                if not slot.start_carla():
                    log(f"slot {slot.idx}: CARLA did not come up")
                    q.put((job, tries))
                    fails_in_row += 1
                    if fails_in_row >= 3:
                        log(f"slot {slot.idx}: giving up")
                        break
                    continue
            ok = slot.run_route(*job)
            with lock:
                stats["done" if ok else "fail"] += 1
            if ok:
                fails_in_row = 0
            else:
                fails_in_row += 1
                slot.stop_carla()  # fresh server after any failure
                if tries + 1 < a.max_tries:
                    q.put((job, tries + 1))
                if fails_in_row >= 5:
                    log(f"slot {slot.idx}: 5 failures in a row, stopping this slot")
                    break
        slot.stop_carla()

    threads = []
    for i, s in enumerate(slots):
        t = threading.Thread(target=worker, args=(s,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(a.stagger)  # do not start all servers at once
    for t in threads:
        t.join()
    left = [j[2] for j in todo if not finished(a.out, j[0], j[2])]
    log(f"finished: {stats['done']} ok attempts, {stats['fail']} failed attempts, {len(left)} routes not done")
    if left:
        log("not done: " + " ".join(left[:50]) + (" ..." if len(left) > 50 else ""))


def clean(results):
    """Same rule as SimLingoBase._route_is_clean."""
    if results["scores"]["score_composed"] < 100.0:
        inf = results["infractions"]
        only_min = results["num_infractions"] == len(inf["min_speed_infractions"]) + len(inf["outside_route_lanes"])
        return results["scores"]["score_route"] > 94.0 and only_min
    return True


def cmd_report(a):
    out = (REPO / a.out) if not os.path.isabs(a.out) else Path(a.out)
    jobs = route_jobs(REPO / a.pairs, set(a.groups.split(",")) if a.groups else None, a.limit_groups)
    CMD = {1: "L", 2: "R", 3: "S", 4: "F", 5: "lcL", 6: "lcR"}
    rows, by_group = [], {}
    for kind, g, name, xml, seed in jobs:
        ds = [d for d in route_dirs(out, kind, name) if os.path.isfile(f"{d}/results.json.gz")]
        if not ds:
            continue
        d = ds[-1]
        n = {s: len(os.listdir(f"{d}/{s}")) if os.path.isdir(f"{d}/{s}") else 0
             for s in ("rgb", "rgb_augmented", "measurements")}
        extra = sorted(set(os.listdir(d)) - {"rgb", "rgb_augmented", "measurements", "results.json.gz"})
        with gzip.open(f"{d}/results.json.gz", "rt") as f:
            res = json.load(f)
        ms = []
        for i in range(n["measurements"]):
            with gzip.open(f"{d}/measurements/{i:04}.json.gz", "rt") as f:
                ms.append(json.load(f))
        seq, first_j = [], None
        for i, m in enumerate(ms):
            c = CMD.get(int(m["command"]), "?") + ("*" if m["junction"] else "")
            if not seq or seq[-1][0] != c:
                seq.append([c, 0])
            seq[-1][1] += 1
            if m["junction"] and first_j is None:
                first_j = i
        x0 = ms[0]["pos_global"][:2] if ms else None
        by_group.setdefault(g, []).append((name, x0, first_j))
        missing = sorted({"ego_matrix", "speed", "command", "next_command", "route", "junction", "target_point",
                          "speed_limit", "augmentation_translation", "augmentation_rotation"} - set(ms[0])) if ms else ["all"]
        rows.append((name, n, res["scores"], res["num_infractions"], clean(res), seq, first_j, extra, missing,
                     ms[-1]["speed"] if ms else None))
    print(f"{len(rows)} collected routes under {out}")
    for name, n, sc, ninf, cl, seq, fj, extra, missing, vend in rows:
        print(f"\n{name}: rgb {n['rgb']} / aug {n['rgb_augmented']} / meas {n['measurements']}  "
              f"score route {sc['score_route']:.1f} composed {sc['score_composed']:.1f} infractions {ninf}  clean {cl}")
        print("  commands (4 Hz, * = junction): " + " ".join(f"{c}x{k}" for c, k in seq))
        print(f"  first junction frame {fj}; end speed {vend:.1f} m/s; other files {extra}; missing fields {missing}")
    print("\ngroups (start distance between routes, first junction frame):")
    for g, items in by_group.items():
        if len(items) < 2:
            continue
        x0 = items[0][1]
        dist = [((it[1][0] - x0[0]) ** 2 + (it[1][1] - x0[1]) ** 2) ** 0.5 for it in items]
        print(f"  {g}: " + ", ".join(f"{it[0].rsplit('_', 1)[1]} d0={d:.1f}m j@{it[2]}" for it, d in zip(items, dist)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--pairs", default="work_dirs/pair_junctions")
        p.add_argument("--out", default="database/e17_pairs")
        p.add_argument("--groups", default="", help="comma separated group names from collect.json")
        p.add_argument("--limit-groups", type=int, default=0)
    r = sub.choices["run"]
    r.add_argument("--garage", default="../carla_garage")
    r.add_argument("--gpu-ids", default="0,0")
    r.add_argument("--base-port", type=int, default=2000)
    r.add_argument("--carla-wait", type=int, default=180, help="seconds to wait for a server")
    r.add_argument("--stagger", type=int, default=20, help="seconds between server starts")
    r.add_argument("--route-timeout", type=int, default=1200, help="seconds per route attempt")
    r.add_argument("--client-timeout", type=int, default=600)
    r.add_argument("--max-tries", type=int, default=3)
    r.add_argument("--no-all-green", dest="all_green", action="store_false")
    a, unknown = ap.parse_known_args()
    if unknown and a.cmd == "run":
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    cmd_run(a) if a.cmd == "run" else cmd_report(a)  # report ignores run-only arguments (sbatch passes both)


if __name__ == "__main__":
    main()
