import {
  Activity,
  Factory
} from "lucide-react";


function Navbar({
  backendOnline
}) {

  return (

    <header className="border-b bg-white">

      <div
        className="
          mx-auto
          flex
          max-w-7xl
          items-center
          justify-between
          px-6
          py-4
        "
      >

        <div
          className="
            flex
            items-center
            gap-3
          "
        >

          <div
            className="
              flex
              h-10
              w-10
              items-center
              justify-center
              rounded-xl
              bg-blue-600
              text-white
            "
          >

            <Factory size={22} />

          </div>

          <div>

            <h1
              className="
                text-lg
                font-bold
                text-slate-900
              "
            >
              DefectVision AI
            </h1>

            <p
              className="
                text-xs
                text-slate-500
              "
            >
              Industrial Quality Inspection
            </p>

          </div>

        </div>


        <div
          className={`
            flex
            items-center
            gap-2
            rounded-full
            px-4
            py-2
            text-sm
            font-semibold

            ${
              backendOnline
                ? "bg-green-50 text-green-700"
                : "bg-red-50 text-red-700"
            }
          `}
        >

          <Activity size={16} />

          {
            backendOnline
              ? "Backend Online"
              : "Backend Offline"
          }

        </div>

      </div>

    </header>

  );
}


export default Navbar;