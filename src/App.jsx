import {
  useCallback,
  useEffect,
  useState
} from "react";

import {
  Camera,
  Cpu,
  ShieldCheck,
  Zap
} from "lucide-react";

import Navbar from
  "./components/Navbar";

import UploadPanel from
  "./components/UploadPanel";

import DetectionResult from
  "./components/DetectionResult";

import Loading from
  "./components/Loading";

import StatsCard from
  "./components/StatsCard";

import {
  getApiErrorMessage,
  getHealth,
  predictAnnotatedImage,
  predictDefects
} from "./services/api";


const MAX_IMAGE_SIZE_BYTES =
  10 * 1024 * 1024;


const ALLOWED_IMAGE_TYPES =
  new Set([
    "image/jpeg",
    "image/png"
  ]);


const DEFAULT_CONFIDENCE =
  0.25;


function App() {

  const [
    backendOnline,
    setBackendOnline
  ] = useState(false);


  const [
    backendInfo,
    setBackendInfo
  ] = useState(null);


  const [
    selectedImage,
    setSelectedImage
  ] = useState(null);


  const [
    result,
    setResult
  ] = useState(null);


  const [
    annotatedImage,
    setAnnotatedImage
  ] = useState(null);


  const [
    loading,
    setLoading
  ] = useState(false);


  const [
    error,
    setError
  ] = useState("");


  // ==========================================
  // BACKEND HEALTH
  // ==========================================

  const checkBackend =
    useCallback(
      async () => {

        try {

          const health =
            await getHealth();


          setBackendInfo(
            health
          );


          const ready =
            health.status === "ok"
            && health.model_loaded === true;


          setBackendOnline(
            ready
          );


          return ready;

        } catch {

          setBackendOnline(
            false
          );

          setBackendInfo(
            null
          );


          return false;
        }

      },
      []
    );


  useEffect(
    () => {

      checkBackend();


      const interval =
        window.setInterval(
          checkBackend,
          10000
        );


      return () => {

        window.clearInterval(
          interval
        );

      };

    },
    [
      checkBackend
    ]
  );


  // ==========================================
  // OBJECT URL CLEANUP
  // ==========================================

  useEffect(
    () => {

      return () => {

        if (
          annotatedImage
        ) {

          URL.revokeObjectURL(
            annotatedImage
          );

        }

      };

    },
    [
      annotatedImage
    ]
  );


  // ==========================================
  // IMAGE SELECTION
  // ==========================================

  const handleImageSelect =
    (file) => {

      setError("");
      setResult(null);
      setAnnotatedImage(null);


      if (
        !ALLOWED_IMAGE_TYPES.has(
          file.type
        )
      ) {

        setSelectedImage(
          null
        );

        setError(
          "Please select a JPG, JPEG or PNG image."
        );

        return;
      }


      if (
        file.size >
        MAX_IMAGE_SIZE_BYTES
      ) {

        setSelectedImage(
          null
        );

        setError(
          "The selected image exceeds the 10 MB upload limit."
        );

        return;
      }


      setSelectedImage(
        file
      );

    };


  // ==========================================
  // DETECTION
  // ==========================================

  const handleDetect =
    async () => {

      if (
        !selectedImage
      ) {

        setError(
          "Please select an image first."
        );

        return;
      }


      const backendReady =
        await checkBackend();


      if (
        !backendReady
      ) {

        setError(
          "The AI backend is not ready. "
          + "Start FastAPI and confirm that "
          + "the optimized model is loaded."
        );

        return;
      }


      setLoading(true);

      setError("");

      setResult(null);

      setAnnotatedImage(null);


      try {

        const prediction =
          await predictDefects(
            selectedImage,
            DEFAULT_CONFIDENCE
          );


        setResult(
          prediction
        );


        const annotatedBlob =
          await predictAnnotatedImage(
            selectedImage,
            DEFAULT_CONFIDENCE
          );


        const imageUrl =
          URL.createObjectURL(
            annotatedBlob
          );


        setAnnotatedImage(
          imageUrl
        );


        setBackendOnline(
          true
        );

      } catch (
        requestError
      ) {

        console.error(
          requestError
        );


        setError(
          getApiErrorMessage(
            requestError
          )
        );


        if (
          !requestError?.response
          ||
          requestError
            ?.response
            ?.status === 503
        ) {

          setBackendOnline(
            false
          );

        }

      } finally {

        setLoading(
          false
        );

      }

    };


  return (

    <div
      className="
        min-h-screen
        bg-slate-50
      "
    >

      <Navbar
        backendOnline={
          backendOnline
        }
      />


      <section
        className="
          border-b
          bg-white
        "
      >

        <div
          className="
            mx-auto
            max-w-7xl
            px-6
            py-12
          "
        >

          <div
            className="
              max-w-3xl
            "
          >

            <div
              className="
                mb-4
                inline-flex
                items-center
                gap-2
                rounded-full
                bg-blue-50
                px-4
                py-2
                text-sm
                font-semibold
                text-blue-700
              "
            >

              <Cpu
                size={16}
              />

              AI Computer Vision

            </div>


            <h2
              className="
                text-4xl
                font-extrabold
                tracking-tight
                text-slate-900
                md:text-5xl
              "
            >

              Real-Time Industrial

              <span
                className="
                  text-blue-600
                "
              >

                {" "}
                Defect Detection

              </span>

            </h2>


            <p
              className="
                mt-5
                max-w-2xl
                text-lg
                leading-8
                text-slate-600
              "
            >

              Automatically inspect industrial
              surfaces and identify defects using
              YOLO-based computer vision with
              confidence scoring, localization,
              and operational quality triage.

            </p>

          </div>

        </div>

      </section>


      <section
        className="
          mx-auto
          max-w-7xl
          px-6
          py-8
        "
      >

        <div
          className="
            grid
            gap-4
            md:grid-cols-3
          "
        >

          <StatsCard

            title="AI Detection"

            value="YOLO"

            description={
              "Six-class steel surface defect detection"
            }

          />


          <StatsCard

            title="Inference"

            value={
              backendInfo?.device
                ?.toUpperCase()
              || "CPU"
            }

            description={
              `Input size ${
                backendInfo
                  ?.image_size
                || 416
              } px`
            }

          />


          <StatsCard

            title="Quality Output"

            value="PASS / REVIEW / REJECT"

            description={
              "Confidence and area-based operational triage"
            }

          />

        </div>

      </section>


      <main
        className="
          mx-auto
          max-w-7xl
          px-6
          pb-12
        "
      >

        <div
          className="
            grid
            gap-6
            lg:grid-cols-2
          "
        >

          <UploadPanel

            selectedImage={
              selectedImage
            }

            onImageSelect={
              handleImageSelect
            }

            onDetect={
              handleDetect
            }

            loading={
              loading
            }

            backendOnline={
              backendOnline
            }

          />


          {
            loading
              ? (

                <div
                  className="
                    rounded-2xl
                    border
                    bg-white
                    shadow-sm
                  "
                >

                  <Loading />

                </div>

              )
              : (

                <DetectionResult

                  result={
                    result
                  }

                  annotatedImage={
                    annotatedImage
                  }

                />

              )
          }

        </div>


        {
          error && (

            <div
              className="
                mt-6
                rounded-xl
                border
                border-red-200
                bg-red-50
                p-4
                text-sm
                font-medium
                text-red-700
              "
            >

              {error}

            </div>

          )
        }


        <section
          className="
            mt-12
          "
        >

          <div
            className="
              mb-6
            "
          >

            <h2
              className="
                text-2xl
                font-bold
              "
            >
              How It Works
            </h2>

            <p
              className="
                mt-1
                text-sm
                text-slate-500
              "
            >
              AI-powered inspection pipeline
            </p>

          </div>


          <div
            className="
              grid
              gap-4
              md:grid-cols-4
            "
          >

            <ProcessCard
              icon={
                <Camera
                  size={28}
                />
              }
              title="01. Capture"
              description={
                "Upload an industrial steel surface image."
              }
            />


            <ProcessCard
              icon={
                <Cpu
                  size={28}
                />
              }
              title="02. AI Analysis"
              description={
                "YOLO detects and localizes surface defects."
              }
            />


            <ProcessCard
              icon={
                <ShieldCheck
                  size={28}
                />
              }
              title="03. Triage"
              description={
                "Detections are evaluated using the quality policy."
              }
            />


            <ProcessCard
              icon={
                <Zap
                  size={28}
                />
              }
              title="04. Decision"
              description={
                "The inspection produces PASS, REVIEW or REJECT."
              }
            />

          </div>

        </section>

      </main>


      <footer
        className="
          border-t
          bg-white
        "
      >

        <div
          className="
            mx-auto
            max-w-7xl
            px-6
            py-6
          "
        >

          <p
            className="
              text-center
              text-sm
              text-slate-500
            "
          >

            AI-Based Real-Time Industrial
            Defect Detection

          </p>

        </div>

      </footer>

    </div>

  );
}


function ProcessCard({
  icon,
  title,
  description
}) {

  return (

    <div
      className="
        rounded-2xl
        border
        bg-white
        p-5
        shadow-sm
      "
    >

      <div
        className="
          text-blue-600
        "
      >
        {icon}
      </div>

      <h3
        className="
          mt-4
          font-bold
        "
      >
        {title}
      </h3>

      <p
        className="
          mt-2
          text-sm
          text-slate-500
        "
      >
        {description}
      </p>

    </div>

  );
}


export default App;
