export default function Loading() {

  return (

    <div className="flex items-center justify-center gap-3 py-10">

      <div className="h-6 w-6 animate-spin rounded-full border-4 border-slate-200 border-t-blue-600" />

      <span className="text-sm text-slate-600">

        AI model is analyzing the image...

      </span>

    </div>

  );
}
