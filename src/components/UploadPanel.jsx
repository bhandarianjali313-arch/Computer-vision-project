import {
  ImagePlus,
  ScanSearch,
  Upload
} from "lucide-react";


function UploadPanel({
  selectedImage,
  onImageSelect,
  onDetect,
  loading,
  backendOnline
}) {

  const handleChange =
    (event) => {

      const file =
        event.target.files?.[0];


      if (
        file
      ) {

        onImageSelect(
          file
        );

      }

    };


  const detectionDisabled =
    !selectedImage
    || loading
    || !backendOnline;


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
          items-center
          gap-3
        "
      >

        <div
          className="
            flex
            h-11
            w-11
            items-center
            justify-center
            rounded-xl
            bg-blue-50
            text-blue-600
          "
        >

          <ImagePlus
            size={22}
          />

        </div>


        <div>

          <h2
            className="
              text-lg
              font-bold
              text-slate-900
            "
          >
            Upload Inspection Image
          </h2>

          <p
            className="
              text-sm
              text-slate-500
            "
          >
            Select an industrial steel
            surface image.
          </p>

        </div>

      </div>


      <label
        className="
          mt-6
          flex
          min-h-[260px]
          cursor-pointer
          flex-col
          items-center
          justify-center
          rounded-2xl
          border-2
          border-dashed
          border-slate-300
          bg-slate-50
          p-8
          text-center
          transition
          hover:border-blue-400
          hover:bg-blue-50
        "
      >

        <Upload
          className="
            text-slate-400
          "
          size={40}
        />


        <p
          className="
            mt-4
            font-semibold
            text-slate-700
          "
        >
          Click to select an image
        </p>


        <p
          className="
            mt-1
            text-sm
            text-slate-500
          "
        >
          JPG, JPEG or PNG — maximum 10 MB
        </p>


        <input
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={
            handleChange
          }
        />

      </label>


      {
        selectedImage && (

          <div
            className="
              mt-4
              rounded-xl
              border
              border-blue-100
              bg-blue-50
              p-4
            "
          >

            <p
              className="
                text-sm
                font-semibold
                text-blue-800
              "
            >
              Selected file
            </p>


            <p
              className="
                mt-1
                break-all
                text-sm
                text-blue-700
              "
            >
              {
                selectedImage.name
              }
            </p>


            <p
              className="
                mt-1
                text-xs
                text-blue-600
              "
            >

              {
                (
                  selectedImage.size
                  / 1024
                  / 1024
                ).toFixed(2)
              }

              {" MB"}

            </p>

          </div>

        )
      }


      {
        !backendOnline && (

          <div
            className="
              mt-4
              rounded-xl
              border
              border-amber-200
              bg-amber-50
              p-3
              text-sm
              text-amber-700
            "
          >

            Detection is disabled until
            the AI backend and model are ready.

          </div>

        )
      }


      <button
        type="button"
        onClick={
          onDetect
        }
        disabled={
          detectionDisabled
        }
        className="
          mt-5
          flex
          w-full
          items-center
          justify-center
          gap-2
          rounded-xl
          bg-blue-600
          px-5
          py-3
          font-semibold
          text-white
          transition
          hover:bg-blue-700
          disabled:cursor-not-allowed
          disabled:bg-slate-300
        "
      >

        <ScanSearch
          size={20}
        />


        {
          loading
            ? "Inspecting..."
            : !backendOnline
              ? "Backend Unavailable"
              : "Detect Defects"
        }

      </button>

    </section>

  );
}


export default UploadPanel;