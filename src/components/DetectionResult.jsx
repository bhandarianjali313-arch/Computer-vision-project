import {
  CheckCircle,
  AlertTriangle,
  Clock,
  Target
} from "lucide-react";


export default function DetectionResult({
  result,
  annotatedImage
}) {

  if (!result) {

    return (

      <div className="flex min-h-[400px] items-center justify-center rounded-2xl border bg-white p-8 text-center shadow-sm">

        <div>

          <Target
            size={45}
            className="mx-auto text-slate-300"
          />

          <h3 className="mt-4 font-semibold text-slate-700">

            Detection results will appear here

          </h3>

          <p className="mt-2 text-sm text-slate-400">

            Upload an inspection image
            to start AI analysis.

          </p>

        </div>

      </div>

    );
  }


  const count =
    result.detection_count || 0;


  const detections =
    result.detections || [];


  return (

    <div className="space-y-5">

      {/* Main result */}

      <div className="rounded-2xl border bg-white p-6 shadow-sm">

        <div className="flex items-center justify-between">

          <div>

            <h2 className="text-lg font-bold">

              Detection Results

            </h2>

            <p className="text-sm text-slate-500">

              AI inspection completed

            </p>

          </div>


          <div
            className={
              count > 0
                ? "rounded-full bg-red-100 px-4 py-2 text-sm font-semibold text-red-700"
                : "rounded-full bg-green-100 px-4 py-2 text-sm font-semibold text-green-700"
            }
          >

            {count > 0
              ? `${count} Defect${count > 1 ? "s" : ""}`
              : "No Defects"}

          </div>

        </div>


        {/* Statistics */}

        <div className="mt-5 grid grid-cols-2 gap-4">

          <div className="rounded-xl bg-slate-50 p-4">

            <div className="flex items-center gap-2 text-slate-500">

              <Target size={16} />

              <span className="text-xs">

                Detections

              </span>

            </div>


            <p className="mt-2 text-xl font-bold">

              {count}

            </p>

          </div>


          <div className="rounded-xl bg-slate-50 p-4">

            <div className="flex items-center gap-2 text-slate-500">

              <Clock size={16} />

              <span className="text-xs">

                Inference Time

              </span>

            </div>


            <p className="mt-2 text-xl font-bold">

              {result.inference_time_ms} ms

            </p>

          </div>

        </div>

      </div>


      {/* Annotated image */}

      {annotatedImage && (

        <div className="rounded-2xl border bg-white p-6 shadow-sm">

          <h3 className="mb-4 font-bold">

            AI Inspection Result

          </h3>


          <div className="overflow-hidden rounded-xl bg-slate-100">

            <img

              src={annotatedImage}

              alt="AI detection result"

              className="max-h-[500px] w-full object-contain"

            />

          </div>

        </div>

      )}


      {/* Detection list */}

      <div className="rounded-2xl border bg-white p-6 shadow-sm">

        <h3 className="mb-4 font-bold">

          Detected Defects

        </h3>


        {detections.length === 0 ? (

          <div className="flex items-center gap-3 rounded-xl bg-green-50 p-4 text-green-700">

            <CheckCircle size={22} />

            <span className="font-medium">

              No defects detected.

            </span>

          </div>

        ) : (

          <div className="space-y-3">

            {detections.map(
              (item, index) => (

                <div

                  key={index}

                  className="rounded-xl border p-4"

                >

                  <div className="flex items-center justify-between">

                    <div className="flex items-center gap-2">

                      <AlertTriangle
                        size={18}
                        className="text-red-500"
                      />

                      <span className="font-semibold">

                        {item.class_name}

                      </span>

                    </div>


                    <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-semibold text-red-600">

                      {(
                        item.confidence *
                        100
                      ).toFixed(1)}%

                    </span>

                  </div>


                  <p className="mt-2 text-xs text-slate-500">

                    Bounding Box:{" "}

                    ({item.bbox.x1},{" "}
                    {item.bbox.y1}) →{" "}
                    ({item.bbox.x2},{" "}
                    {item.bbox.y2})

                  </p>

                </div>

              )
            )}

          </div>

        )}

      </div>

    </div>

  );
}
