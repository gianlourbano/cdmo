import argparse
import threading
import time
import sys
from datetime import timedelta
from minizinc import Instance, Model, Solver

class Timer:
    def __init__(self, duration=300):  # 300 seconds = 5 minutes
        self.duration = duration
        self.start_time = None
        self.running = False
        self.timer_thread = None
        
    def start(self):
        self.start_time = time.time()
        self.running = True
        self.timer_thread = threading.Thread(target=self._run_timer, daemon=True)
        self.timer_thread.start()
        
    def stop(self):
        self.running = False
        
    def _run_timer(self):
        while self.running:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.duration - elapsed)
            
            mins, secs = divmod(int(remaining), 60)
            print(f"\rTime remaining: {mins:02d}:{secs:02d}", end="", flush=True)
            
            if remaining <= 0:
                print("\n⏰ Time limit (5 minutes) reached!")
                break
                
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Solve Sports Tournament Scheduling problem")
    parser.add_argument("n", type=int, help="Number of teams (must be even)")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout in seconds (default: 300 = 5 minutes)")
    
    args = parser.parse_args()
    
    if args.n % 2 != 0:
        print("Error: Number of teams must be even!")
        sys.exit(1)
    
    print(f"🏆 Solving Sports Tournament Scheduling for {args.n} teams")
    print(f"⏱️  Timeout set to {args.timeout} seconds")
    print("-" * 50)
    
    model = Model("instances/test.mzn")
    solver = Solver.lookup("gecode")
    instance = Instance(solver, model)
    instance["n"] = args.n
    
    # Start the timer
    timer = Timer(args.timeout)
    timer.start()
    
    try:
        result = instance.solve(timeout=timedelta(seconds=args.timeout))
        timer.stop()
        print("\n" + "=" * 50)
        
        if result == "None":
            print("❌ No solution found within the time limit!")
            print("💡 Try increasing the timeout or reducing the problem size.")
        else:
            print("✅ Solution found!")
            print("=" * 50)
            print(result)
    except Exception as e:
        timer.stop()
        print(f"\n❌ Error occurred: {e}")
    finally:
        print("\n🏁 Execution completed.")

if __name__ == "__main__":
    main()
