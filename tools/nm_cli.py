#!/usr/bin/env python3
"""Small VM CLI. Author on Windows; build, run and test only on Linux."""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / os.environ.get("NM_BUILD_DIR", "build")
IMAGE = "neurominds-toolchain:0.1"
RUN_LOG = None


def run(command, log=None, env=None):
    log = log or RUN_LOG
    command = list(map(str, command))
    print("+ " + " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    with open(log, "a", encoding="utf-8") as stream:
        stream.write("\n+ " + " ".join(command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def build(args=()):
    p = argparse.ArgumentParser(prog="./nm build")
    p.add_argument("--scalar-only", action="store_true")
    p.add_argument("--sanitizers", action="store_true")
    p.add_argument("--config", default=str(ROOT / "configs/prototype.json"))
    p.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", "2")))
    a = p.parse_args(args)
    if a.jobs < 1:
        p.error("--jobs must be positive")
    command = ["cmake", "-S", ROOT, "-B", BUILD, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
               "-DPython3_EXECUTABLE=" + sys.executable, "-DNM_CONFIG=" + str(Path(a.config).resolve()),
               "-DNM_ENABLE_MLIR=" + ("OFF" if a.scalar_only else "ON"),
               "-DNM_SANITIZERS=" + ("ON" if a.sanitizers else "OFF")]
    if not a.scalar_only:
        prefix = Path(os.environ["TORCH_MLIR_BUILD"])
        command += ["-DNM_LLVM_BIN=" + str(prefix / "bin"), "-DMLIR_DIR=" + str(prefix / "lib/cmake/mlir"),
                    "-DLLVM_DIR=" + str(prefix / "lib/cmake/llvm")]
        # Reuse the compiler that built the user's already working LLVM tree.
        # This avoids accidentally selecting Ubuntu 20.04's older default GCC.
        upstream_cache = prefix / "CMakeCache.txt"
        if upstream_cache.exists():
            lines = upstream_cache.read_text().splitlines()
            for variable, environment in (("CMAKE_C_COMPILER", "CC"), ("CMAKE_CXX_COMPILER", "CXX")):
                if not os.environ.get(environment):
                    for line in lines:
                        if line.startswith(variable + ":") and "=" in line:
                            compiler = line.split("=", 1)[1]
                            if Path(compiler).is_file():
                                command.append("-D" + variable + "=" + compiler)
                            break
    run(command)
    run(["cmake", "--build", BUILD, "--parallel", a.jobs])


def doctor():
    directory = ROOT / "results"
    directory.mkdir(exist_ok=True)
    report = {"platform": platform.platform(), "python": sys.version, "python_executable": sys.executable,
              "logical_cores": os.cpu_count(), "toolchain_source": os.environ.get("TORCH_MLIR_SRC"),
              "toolchain_build": os.environ.get("TORCH_MLIR_BUILD"), "commands": {}}
    if hasattr(os, "sched_getaffinity"):
        report["allowed_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    report["thread_environment"] = {key: os.environ.get(key) for key in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    commands = [["uname", "-a"], ["lscpu"], ["free", "-h"], ["gcc", "--version"], ["clang", "--version"],
                ["cmake", "--version"], ["mlir-opt", "--version"], ["opt", "--version"], ["llc", "--version"], ["nvidia-smi"],
                [sys.executable, "-m", "pip", "freeze"],
                [sys.executable, "-c", "import torch, torch_mlir; print('torch:',torch.__version__); print('torch_mlir:',torch_mlir.__file__)"]]
    for command in commands:
        if not shutil.which(command[0]):
            value = "not installed/on PATH"
        else:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            value = process.stdout
            if process.returncode:
                value += "\nexit=" + str(process.returncode)
        report["commands"][" ".join(command)] = value
        print("\n" + " ".join(command) + "\n" + value)
    for filename in ("/etc/os-release", "/proc/meminfo"):
        report[filename] = Path(filename).read_text() if Path(filename).exists() else "unavailable"
    for key in ("TORCH_MLIR_SRC",):
        source = Path(os.environ[key])
        for name, path in (("torch_mlir_revision", source), ("llvm_revision", source / "externals/llvm-project")):
            process = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            report[name] = process.stdout.strip() if process.returncode == 0 else "unavailable"
    report["telemetry"] = {"cpufreq_path_present": Path("/sys/devices/system/cpu/cpu0/cpufreq").exists(),
                           "powercap_path_present": Path("/sys/class/powercap").exists(),
                           "thermal_class_present": Path("/sys/class/thermal").exists(),
                           "perf_executable": shutil.which("perf"),
                           "note": "Presence does not prove guest access or reliable counters. No thermal, power, energy or perf-counter measurements are claimed."}
    (directory / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved results/environment.json")


def binary():
    result = BUILD / "neurominds_demo"
    if not result.exists():
        raise RuntimeError("Run ./nm build first")
    return result


def tests():
    if not (BUILD / "CTestTestfile.cmake").exists():
        raise RuntimeError("Run ./nm build first")
    # CTest output + exit code are preserved; no claimed success without execution.
    run(["ctest", "--test-dir", BUILD, "--output-on-failure"])


def all_steps(args, sweep=False):
    global RUN_LOG
    p = argparse.ArgumentParser(prog="./nm " + ("sweep" if sweep else "all"))
    p.add_argument("--config", default=str(ROOT / "configs/prototype.json"))
    p.add_argument("--batch", type=int)
    p.add_argument("--input", type=int)
    p.add_argument("--output", type=int)
    p.add_argument("--group", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--calibration-samples", type=int, default=64)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--sampled", action="store_true")
    p.add_argument("--backend", choices=("mlir", "scalar"), default="mlir")
    p.add_argument("--results", default="results/run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S"))
    a = p.parse_args(args)
    defaults = json.loads(Path(a.config).read_text())["defaults"]
    for key in ("batch", "input", "output", "group", "seed"):
        if getattr(a, key) is None:
            setattr(a, key, defaults[key])
    target = Path(a.results).resolve()
    target.mkdir(parents=True, exist_ok=False)
    RUN_LOG = target / "run.log"
    doctor()
    build_args = ["--config", a.config]
    if a.backend == "scalar":
        build_args += ["--scalar-only"]
    build(build_args)
    tests()
    for batch in ([1, 4, 8, 16] if sweep else [a.batch]):
        destination = target / ("batch-" + str(batch)) if sweep else target
        destination.mkdir(parents=True, exist_ok=True)
        log = destination / "run.log"
        common = ["--batch", batch, "--input", a.input, "--output", a.output, "--group", a.group,
                  "--seed", a.seed, "--backend", a.backend, "--results", destination,
                  "--policy", destination / "policy.nm", "--warmup", a.warmup, "--repeats", a.repeats]
        if a.sampled:
            common += ["--sampled"]
        run([binary(), "calibrate", *common, "--samples", a.calibration_samples], log)
        run([binary(), "demo", *common], log)
        run([binary(), "bench", *common, "--samples", a.samples], log)
        run([sys.executable, ROOT / "tools/summarize.py", destination], log)
        provenance = destination / "compiler"
        provenance.mkdir(exist_ok=True)
        for filename in ("build_manifest.json", "compiler_manifest.json", "linear.torch.mlir", "linear.linalg.mlir",
                         "executors.mlir", "executors.prepared.mlir", "executors.llvm.mlir", "executors.ll"):
            source = BUILD / "generated" / filename
            if source.exists():
                shutil.copyfile(source, provenance / filename)
        shutil.copyfile(ROOT / "results/environment.json", destination / "environment.json")
        shutil.copyfile(BUILD / "CMakeCache.txt", provenance / "CMakeCache.txt")
        if (BUILD / "compile_commands.json").exists():
            shutil.copyfile(BUILD / "compile_commands.json", provenance / "compile_commands.json")
    print("\nCompleted VM workflow. Results: " + str(target))


def container(command, args):
    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed/on PATH. Reuse your existing ~/tools/torch-mlir natively, or install Docker on the VM first; see docs/UBUNTU_20_04.md")
    if command == "setup":
        run(["docker", "build", "--progress=plain", "-t", IMAGE, "."])
    else:
        run(["docker", "run", "--rm", "--user", str(os.getuid()) + ":" + str(os.getgid()),
             "-e", "XDG_CACHE_HOME=/tmp/nm-cache", "-e", "NM_BUILD_DIR=build-container", "-v", str(ROOT) + ":/work", "-w", "/work",
             IMAGE, "bash", "./nm", command, *args])


def main():
    if sys.platform != "linux":
        raise RuntimeError("Builds and experiments are intended for your Ubuntu VM. This CLI does not run tests on Windows.")
    args = sys.argv[1:]
    use_container = "--container" in args
    args = [arg for arg in args if arg != "--container"]
    command = args.pop(0) if args else "help"
    if use_container:
        container(command, args)
        return
    os.chdir(ROOT)
    if command in ("help", "--help", "-h"):
        print(__doc__ + "\n\n./nm doctor | setup | build | test | calibrate | demo | run | bench | summarize | all | sweep | collect\n"
              "Append --container to use the optional Ubuntu 24.04 compiler image on the Ubuntu 20.04 host.\n"
              "Existing toolchain default: ~/tools/torch-mlir (same layout as mlir-quant-linear).\n"
              "Examples: ./nm all; ./nm all --sampled; ./nm sweep; ./nm run --variant a8; ./nm build --scalar-only")
    elif command == "build":
        build(args)
    elif command == "setup":
        run(["bash", ROOT / "scripts/setup_toolchain.sh", *args])
    elif command in ("doctor", "env"):
        doctor()
    elif command == "test":
        tests()
    elif command in ("all", "sweep"):
        all_steps(args, command == "sweep")
    elif command == "summarize":
        run([sys.executable, ROOT / "tools/summarize.py", *(args or [ROOT / "results"])])
    elif command == "collect":
        target = Path(args[0] if args else "results").resolve()
        if ROOT not in target.parents or not target.is_dir():
            raise ValueError("collect requires a results directory inside this repository")
        archive = ROOT / "results" / ("diagnostics-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S") + ".tar.gz")
        with tarfile.open(archive, "w:gz") as output:
            for file in target.rglob("*"):
                if file.is_file() and file.suffix in (".json", ".csv", ".log", ".txt", ".md", ".nm", ".mlir", ".ll"):
                    output.add(file, arcname=str(file.relative_to(ROOT)))
        print(archive)
    else:
        # C++ parser validates the command and every option.
        run([binary(), command, *args])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("FAILED (exit %s): %s" % (error.returncode, error.cmd), file=sys.stderr)
        sys.exit(error.returncode)
    except (RuntimeError, ValueError, OSError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
