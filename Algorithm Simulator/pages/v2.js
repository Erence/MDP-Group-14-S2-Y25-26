import Link from "next/link";
import SimulatorV2 from "@/components/SimulatorV2";

export default function V2Page() {
  return (
    <main className="min-h-screen p-4">
      <div className="flex justify-center mb-4">
        <div className="join">
          <Link href="/" className="btn btn-sm btn-outline join-item">
            Grid Simulator
          </Link>
          <Link href="/v2" className="btn btn-sm btn-primary join-item">
            Free-Range Simulator
          </Link>
        </div>
      </div>
      <SimulatorV2 />
    </main>
  );
}
