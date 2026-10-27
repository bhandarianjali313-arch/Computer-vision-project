import {
  AlertTriangle,
  CheckCircle2,
  Search,
  ShieldAlert
} from "lucide-react";


function DecisionBadge({
  decision
}) {

  if (decision === "PASS") {

    return (

      <span
        className="
          inline-flex
          items-center
          gap-2
          rounded-full
          bg-green-100
          px-3
          py-1
          text-sm
          font-bold
          text-green-700
        "
      >

        <CheckCircle2 size={16} />

        PASS

      </span>

    );

  }


  if (decision === "REJECT") {

    return (

      <span
        className="
          inline-flex
          items-center
          gap-2
          rounded-full
          bg-red-100
          px-3
          py-1
          text-sm
          font-bold
          text-red-700
        "
      >

        <ShieldAlert size={16} />

        REJECT

      </span>

    );

  }


  return (

    <span
      className="
        inline-flex
        items-center
        gap-2
        rounded-full
        bg-amber-100
        px-3
        py-1
        text-sm
        font-bold
        text-amber-700
      "
    >

      <AlertTriangle size={16} />

      {
        decision || "REVIEW"
      }

    </span>

  );

}


function DetectionResult({
  result,
  annotatedImage
}) {

  if (!result) {

    return (

      <section
        className="
          flex
          min-h-[500px]
          flex-col
          items-center
          justify-center
          rounded-2xl
          border
          border-slate-200
          bg-white
          p-8
          text-center
          shadow-sm
        "
      >

        <div
          className="
            flex
            h-16
            w-16
            items-center
            justify-center
            rounded-2xl
            bg-slate-100
            text-slate-400
          "
        >
          <Search size={32} />
        </div>

        <h2
          className="
            mt-5
            text-lg
            font-bold
            text-slate-800
          "
        >
          Inspection Results
        </h2>

        <p
          className="
            mt-2
            max-w-sm
            text-sm
            text-slate-500
          "
        >
          Upload an industrial surface
          image and run defect detection
          to view the inspection result.
        </p>

      </section>

    );

  }


  const detections =
    result.detections || [];

  const triage =
    result.triage_counts || {};


  return (

    <section
      className="
        rounded-2xl
        border
        border-slate-200
        bg-white
        p-6
        shadow-sm
      "
    >

      <div
        className="
          flex
          flex-wrap
          items-start
          justify-between
          gap-4
        "
      >

        <div>

          <p
            className="
              text-sm
              font-medium
              text-slate-500
            "
          >
            Quality Decision
          </p>

          <div className="mt-2">

            <DecisionBadge
              decision={
                result.quality_decision
              }
            />

          </div>

        </div>


        <div className="text-right">

          <p
            className="
              text-sm
              text-slate-500
            "
          >
            Inference Time
          </p>

          <p
            className="
              text-lg
              font-bold
              text-slate-900
            "
          >
            {
              result.inference_time_ms
              ?? 0
            } ms
          </p>

        </div>

      </div>


      <div
        className="
          mt-6
          grid
          grid-cols-2
          gap-3
          sm:grid-cols-4
        "
      >

        <ResultStat
          title="Defects"
          value={
            result.detection_count
            ?? detections.length
          }
        />

        <ResultStat
          title="Low"
          value={
            triage.LOW ?? 0
          }
        />

        <ResultStat
          title="Medium"
          value={
            triage.MEDIUM ?? 0
          }
        />

        <ResultStat
          title="High"
          value={
            triage.HIGH ?? 0
          }
        />

      </div>


      {
        annotatedImage && (

          <div className="mt-6">

            <p
              className="
                mb-3
                text-sm
                font-semibold
                text-slate-700
              "
            >
              Annotated Inspection
            </p>

            <img
              src={annotatedImage}
              alt="Detected industrial defects"
              className="
                w-full
                rounded-xl
                border
                border-slate-200
                object-contain
              "
            />

          </div>

        )
      }


      <div className="mt-6">

        <div
          className="
            flex
            items-center
            justify-between
          "
        >

          <h3
            className="
              font-bold
              text-slate-900
            "
          >
            Detected Defects
          </h3>

          <span
            className="
              rounded-full
              bg-slate-100
              px-3
              py-1
              text-xs
              font-semibold
              text-slate-600
            "
          >
            {detections.length} detections
          </span>

        </div>


        {
          detections.length === 0
            ? (

              <div
                className="
                  mt-4
                  rounded-xl
                  bg-green-50
                  p-4
                  text-sm
                  text-green-700
                "
              >
                No defects remained after
                quality post-processing.
              </div>

            )
            : (

              <div
                className="
                  mt-4
                  max-h-64
                  space-y-3
                  overflow-y-auto
                "
              >

                {
                  detections.map(
                    (
                      detection,
                      index
                    ) => (

                      <div
                        key={
                          `${detection.class_name}-${index}`
                        }
                        className="
                          rounded-xl
                          border
                          border-slate-200
                          p-4
                        "
                      >

                        <div
                          className="
                            flex
                            items-center
                            justify-between
                            gap-3
                          "
                        >

                          <span
                            className="
                              font-semibold
                              capitalize
                              text-slate-900
                            "
                          >
                            {
                              detection
                                .class_name
                                ?.replaceAll(
                                  "_",
                                  " "
                                )
                            }
                          </span>

                          <span
                            className="
                              text-sm
                              font-semibold
                              text-blue-600
                            "
                          >
                            {
                              (
                                (
                                  detection
                                    .confidence
                                  ?? 0
                                )
                                * 100
                              ).toFixed(1)
                            }%
                          </span>

                        </div>


                        {
                          detection
                            .triage_level && (

                            <p
                              className="
                                mt-2
                                text-xs
                                font-medium
                                text-slate-500
                              "
                            >
                              Triage:
                              {" "}
                              {
                                detection
                                  .triage_level
                              }
                            </p>

                          )
                        }

                      </div>

                    )
                  )
                }

              </div>

            )
        }

      </div>


      {
        result.decision_reasons
        ?.length > 0 && (

          <div
            className="
              mt-6
              rounded-xl
              bg-slate-50
              p-4
            "
          >

            <p
              className="
                text-sm
                font-semibold
                text-slate-700
              "
            >
              Decision reasons
            </p>

            <ul
              className="
                mt-2
                list-disc
                space-y-1
                pl-5
                text-sm
                text-slate-600
              "
            >

              {
                result
                  .decision_reasons
                  .map(
                    (
                      reason,
                      index
                    ) => (

                      <li key={index}>
                        {reason}
                      </li>

                    )
                  )
              }

            </ul>

          </div>

        )
      }

    </section>

  );

}


function ResultStat({
  title,
  value
}) {

  return (

    <div
      className="
        rounded-xl
        bg-slate-50
        p-3
        text-center
      "
    >

      <p
        className="
          text-xl
          font-bold
          text-slate-900
        "
      >
        {value}
      </p>

      <p
        className="
          text-xs
          font-medium
          text-slate-500
        "
      >
        {title}
      </p>

    </div>

  );

}


export default DetectionResult;