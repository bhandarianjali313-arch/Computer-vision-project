export default function StatsCard({
  title,
  value,
  description
}) {

  return (

    <div className="rounded-2xl border bg-white p-5 shadow-sm">

      <p className="text-sm font-medium text-slate-500">

        {title}

      </p>


      <h3 className="mt-2 text-2xl font-bold text-slate-900">

        {value}

      </h3>


      {description && (

        <p className="mt-1 text-xs text-slate-500">

          {description}

        </p>

      )}

    </div>

  );
}
