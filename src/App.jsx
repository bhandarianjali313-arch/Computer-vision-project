import {
  useEffect,
  useState
} from "react";

import axios from "axios";

import {
  Cpu,
  Camera,
  ShieldCheck,
  Zap
} from "lucide-react";

import Navbar from "./components/Navbar";

import UploadPanel from "./components/UploadPanel";

import DetectionResult from "./components/DetectionResult";

import Loading from "./components/Loading";

import StatsCard from "./components/StatsCard";


const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";


function App() {

  const [
    backendOnline,
    setBackendOnline
  ] = useState(false);


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
  // CHECK BACKEND
  // ==========================================

  useEffect(() => {

    checkBackend();

  }, []);


  const checkBackend =
    async () => {

      try {

        await axios.get(
          `${API_URL}/health`
        );

        setBackendOnline(true);

      } catch {

        setBackendOnline(false);

      }

    };


  // ==========================================
  // SELECT IMAGE
  // ==========================================

  const handleImageSelect =
    (file) => {

      setSelectedImage(file);

      setResult(null);

      setAnnotatedImage(null);

      setError("");

    };


  // ==========================================
  // DETECT DEFECTS
  // ==========================================

  const handleDetect =
    async () => {

      if (!selectedImage) {

        setError(
          "Please select an image first."
        );

        return;

      }


      setLoading(true);

      setError("");

      setResult(null);

      setAnnotatedImage(null);


      try {

        const formData =
          new FormData();


        formData.append(
          "file",
          selectedImage
        );


        // --------------------------------------
        // JSON Detection
        // --------------------------------------

        const response =
          await axios.post(

            `${API_URL}/predict`,

            formData,

            {

              headers: {

                "Content-Type":
                  "multipart/form-data"

              }

            }

          );


        setResult(
          response.data
        );


        // --------------------------------------
        // Annotated Image
        // --------------------------------------

        const imageResponse =
          await axios.post(

            `${API_URL}/predict/image`,

            formData,

            {

              responseType:
                "blob",

              headers: {

                "Content-Type":
                  "multipart/form-data"

              }

            }

          );


        const imageUrl =
          URL.createObjectURL(
            imageResponse.data
          );


        setAnnotatedImage(
          imageUrl
        );


      } catch (error) {

        console.error(
          error
        );


        if (
          error.response?.data
            ?.detail
        ) {

          setError(
            error.response.data.detail
          );

        } else {

          setError(
            "Unable to connect to the AI backend. Make sure FastAPI is running."
          );

        }

      } finally {

        setLoading(false);

      }

    };


  return (

    <div className="min-h-screen bg-slate-50">

      {/* ======================================
          NAVBAR
      ====================================== */}

      <Navbar
        backendOnline={
          backendOnline
        }
      />


      {/* ======================================
          HERO
      ====================================== */}

      <section className="border-b bg-white">

        <div className="mx-auto max-w-7xl px-6 py-12">

          <div className="max-w-3xl">

            <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-blue-50 px-4 py-2 text-sm font-semibold text-blue-700">

              <Cpu size={16} />

              AI Computer Vision

            </div>


            <h2 className="text-4xl font-extrabold tracking-tight text-slate-900 md:text-5xl">

              Real-Time Industrial

              <span className="text-blue-600">

                {" "}Defect Detection

              </span>

            </h2>


            <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">

              Automatically inspect industrial
              surfaces and identify defects using
              YOLO-based computer vision with
              confidence scoring and bounding-box
              localization.

            </p>

          </div>

        </div>

      </section>


      {/* ======================================
          FEATURES
      ====================================== */}

      <section className="mx-auto max-w-7xl px-6 py-8">

        <div className="grid gap-4 md:grid-cols-3">

          <StatsCard

            title="AI Detection"

            value="YOLO"

            description="Deep-learning object detection"

          />


          <StatsCard

            title="Processing"

            value="Real-Time"

            description="Fast image inference"

          />


          <StatsCard

            title="Output"

            value="BBox + Score"

            description="Defect location and confidence"

          />

        </div>

      </section>


      {/* ======================================
          MAIN DASHBOARD
      ====================================== */}

      <main className="mx-auto max-w-7xl px-6 pb-12">

        <div className="grid gap-6 lg:grid-cols-2">


          {/* Upload */}

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

          />


          {/* Results */}

          {loading ? (

            <div className="rounded-2xl border bg-white shadow-sm">

              <Loading />

            </div>

          ) : (

            <DetectionResult

              result={
                result
              }

              annotatedImage={
                annotatedImage
              }

            />

          )}

        </div>


        {/* ======================================
            ERROR
        ====================================== */}

        {error && (

          <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm font-medium text-red-700">

            {error}

          </div>

        )}


        {/* ======================================
            HOW IT WORKS
        ====================================== */}

        <section className="mt-12">

          <div className="mb-6">

            <h2 className="text-2xl font-bold">

              How It Works

            </h2>

            <p className="mt-1 text-sm text-slate-500">

              AI-powered inspection pipeline

            </p>

          </div>


          <div className="grid gap-4 md:grid-cols-4">


            <div className="rounded-2xl border bg-white p-5 shadow-sm">

              <Camera
                className="text-blue-600"
                size={28}
              />

              <h3 className="mt-4 font-bold">

                01. Capture

              </h3>

              <p className="mt-2 text-sm text-slate-500">

                Capture an industrial product
                or surface image.

              </p>

            </div>


            <div className="rounded-2xl border bg-white p-5 shadow-sm">

              <Cpu
                className="text-blue-600"
                size={28}
              />

              <h3 className="mt-4 font-bold">

                02. AI Analysis

              </h3>

              <p className="mt-2 text-sm text-slate-500">

                YOLO processes the image and
                identifies possible defects.

              </p>

            </div>


            <div className="rounded-2xl border bg-white p-5 shadow-sm">

              <ShieldCheck
                className="text-blue-600"
                size={28}
              />

              <h3 className="mt-4 font-bold">

                03. Inspection

              </h3>

              <p className="mt-2 text-sm text-slate-500">

                Defects are localized with
                bounding boxes and confidence.

              </p>

            </div>


            <div className="rounded-2xl border bg-white p-5 shadow-sm">

              <Zap
                className="text-blue-600"
                size={28}
              />

              <h3 className="mt-4 font-bold">

                04. Decision

              </h3>

              <p className="mt-2 text-sm text-slate-500">

                Results can be used for
                quality-control decisions.

              </p>

            </div>


          </div>

        </section>

      </main>


      {/* ======================================
          FOOTER
      ====================================== */}

      <footer className="border-t bg-white">

        <div className="mx-auto max-w-7xl px-6 py-6">

          <p className="text-center text-sm text-slate-500">

            AI-Based Real-Time Industrial
            Defect Detection

          </p>

        </div>

      </footer>

    </div>

  );
}


export default App;
