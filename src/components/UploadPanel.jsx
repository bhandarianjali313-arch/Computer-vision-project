import {
  Upload,
  Image as ImageIcon
} from "lucide-react";


export default function UploadPanel({

  selectedImage,

  onImageSelect,

  onDetect,

  loading

}) {

  const handleFileChange = (
    event
  ) => {

    const file =
      event.target.files?.[0];

    if (!file) return;

    onImageSelect(file);
  };


  return (

    <div className="rounded-2xl border bg-white p-6 shadow-sm">

      <div className="mb-5">

        <h2 className="text-lg font-bold">

          Upload Inspection Image

        </h2>

        <p className="mt-1 text-sm text-slate-500">

          Upload a product or surface image
          for AI defect detection.

        </p>

      </div>


      {/* Upload area */}

      <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 px-6 py-10 transition hover:border-blue-500 hover:bg-blue-50">

        <Upload
          size={35}
          className="text-blue-600"
        />


        <p className="mt-3 font-semibold">

          Choose inspection image

        </p>


        <p className="mt-1 text-xs text-slate-500">

          PNG, JPG or WEBP

        </p>


        <input

          type="file"

          accept="image/png,image/jpeg,image/webp"

          onChange={
            handleFileChange
          }

          className="hidden"

        />

      </label>


      {/* Selected image */}

      {selectedImage && (

        <div className="mt-5">

          <p className="mb-2 text-sm font-medium">

            Selected Image

          </p>


          <div className="overflow-hidden rounded-xl border">

            <img

              src={
                URL.createObjectURL(
                  selectedImage
                )
              }

              alt="Selected inspection"

              className="max-h-80 w-full object-contain bg-slate-100"

            />

          </div>


          <p className="mt-2 flex items-center gap-2 text-xs text-slate-500">

            <ImageIcon size={14} />

            {selectedImage.name}

          </p>


          <button

            onClick={onDetect}

            disabled={loading}

            className="mt-5 w-full rounded-xl bg-blue-600 px-5 py-3 font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"

          >

            {loading
              ? "Analyzing..."
              : "Detect Defects"}

          </button>

        </div>

      )}

    </div>

  );
}
