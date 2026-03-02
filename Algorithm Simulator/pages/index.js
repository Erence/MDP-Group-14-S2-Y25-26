import Link from "next/link";
import Simulator from "@/components/Simulator";

export default function HomePage() {
  return (
    <main className="min-h-screen p-4">
      <div className="flex justify-center mb-4">
        <div className="join">
          <Link href="/" className="btn btn-sm btn-primary join-item">
            Grid Simulator
          </Link>
          <Link href="/v2" className="btn btn-sm btn-outline join-item">
            Free-Range Simulator
          </Link>
        </div>
      </div>
      <Simulator />
    </main>
  );
}
