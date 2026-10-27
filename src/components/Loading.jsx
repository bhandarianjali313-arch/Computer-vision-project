function Loading() {

  return (

    <div
      className="
        flex
        min-h-[420px]
        flex-col
        items-center
        justify-center
        p-8
        text-center
      "
    >

      <div
        className="
          h-12
          w-12
          animate-spin
          rounded-full
          border-4
          border-slate-200
          border-t-blue-600
        "
      />

      <h3
        className="
          mt-5
          text-lg
          font-semibold
          text-slate-900
        "
      >
        Inspecting Surface
      </h3>

      <p
        className="
          mt-2
          max-w-sm
          text-sm
          text-slate-500
        "
      >
        The YOLO model is analysing the
        image and applying the quality
        decision policy.
      </p>

    </div>

  );
}


export default Loading;