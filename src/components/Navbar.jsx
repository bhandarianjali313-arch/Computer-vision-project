import {
  ShieldCheck,
  Activity
} from "lucide-react";


export default function Navbar({
  backendOnline
}) {

  return (

    <nav className="border-b bg-white">

      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">

        {/* Logo */}

        <div className="flex items-center gap-3">

          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white">

            <ShieldCheck
              size={23}
            />

          </div>


          <div>

            <h1 className="text-lg font-bold text-slate-900">

              Industrial Defect Detection

            </h1>

            <p className="text-xs text-slate-500">

              AI-Powered Computer Vision

            </p>

          </div>

        </div>


        {/* Backend status */}

        <div className="flex items-center gap-2 rounded-full border px-3 py-2">

          <Activity
            size={15}
            className={
              backendOnline
                ? "text-green-500"
                : "text-red-500"
            }
          />

          <span className="text-sm font-medium">

            {backendOnline
              ? "Backend Online"
              : "Backend Offline"}

          </span>

        </div>

      </div>

    </nav>

  );
}
