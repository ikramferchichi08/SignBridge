import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BarChart3,
  Database,
  FlaskConical,
  Hand,
  Home,
  Info,
  Play,
  Search,
  ShieldCheck,
  Sparkles,
  Video,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const API = "http://localhost:8000";

async function getJson(path) {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(`Erreur API: ${response.status}`);
  return response.json();
}

function App() {
  const [page, setPage] = useState("overview");
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Hand size={22} /></div>
          <div>
            <strong>SignBridge</strong>
            <span>Sign recognition lab</span>
          </div>
        </div>

        <nav>
          <NavButton page="overview" active={page} setPage={setPage} icon={Home} label="Vue d'ensemble" />
          <NavButton page="dataset" active={page} setPage={setPage} icon={Database} label="Dataset" />
          <NavButton page="experiments" active={page} setPage={setPage} icon={FlaskConical} label="Expériences" />
          <NavButton page="live" active={page} setPage={setPage} icon={Video} label="Démo live" />
        </nav>

        <div className="sidebar-status">
          <div><span className="status-dot" /> Système opérationnel</div>
          <small>Version 1.0 · ML local</small>
        </div>
      </aside>

      <main>
        {page === "overview" && <Overview />}
        {page === "dataset" && <Dataset />}
        {page === "experiments" && <Experiments />}
        {page === "live" && <LiveDemo />}
      </main>
    </div>
  );
}

function NavButton({ page, active, setPage, icon: Icon, label }) {
  return (
    <button className={active === page ? "nav-button active" : "nav-button"} onClick={() => setPage(page)}>
      <Icon size={18} />
      <span>{label}</span>
    </button>
  );
}

function PageHeader({ eyebrow, title, description }) {
  return (
    <header className="page-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <button className="icon-button" aria-label="Informations"><Info size={18} /></button>
    </header>
  );
}

function Metric({ label, value, detail, accent = false }) {
  return (
    <article className={accent ? "metric-card accent" : "metric-card"}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}

function Overview() {
  const [data, setData] = useState(null);
  useEffect(() => { getJson("/api/overview").then(setData).catch(console.error); }, []);
  if (!data) return <Loading />;

  const best = data.best_model;
  return (
    <>
      <PageHeader
        eyebrow="Tableau de bord"
        title="Vue d'ensemble"
        description="Suivez les données, les expériences et l'avancée du prototype SignBridge."
      />

      <section className="hero-card">
        <div className="hero-copy">
          <span className="hero-eyebrow"><Sparkles size={14} /> Modèle actuel</span>
          <h2>Reconnaître les signes à partir des mouvements.</h2>
          <p>
            SignBridge transforme les positions du corps et des mains en séquences
            de 32 landmarks pour apprendre les gestes isolés.
          </p>
          <div className="hero-tags">
            <span>100 signes</span>
            <span>MediaPipe</span>
            <span>Signer-aware</span>
          </div>
        </div>
        <div className="hero-score">
          <span>Top-1 moyen</span>
          <strong>{(best.top1_mean * 100).toFixed(1)}<small>%</small></strong>
          <div className="score-track"><div style={{ width: `${best.top1_mean * 100}%` }} /></div>
          <small>Meilleure expérience · Transformer</small>
        </div>
      </section>

      <section className="metric-grid overview-metrics">
        <Metric label="Vidéos WLASL" value={data.videos.toLocaleString("fr-FR")} detail="Dataset signé" />
        <Metric label="Signes" value={data.signs} detail="Vocabulaire isolé" />
        <Metric label="Signers" value={data.signers} detail="Personnes distinctes" />
        <Metric label="Meilleur Top-1" value={`${(best.top1_mean * 100).toFixed(1)}%`} detail="Validation croisée" accent />
      </section>

      <section className="two-column">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">Pipeline</span>
              <h2>Workflow du projet</h2>
            </div>
            <ShieldCheck size={20} />
          </div>
          <div className="workflow">
            {data.workflow.map((item, index) => (
              <div className="workflow-row" key={item.title}>
                <div className={item.status === "done" ? "workflow-index done" : "workflow-index"}>{index + 1}</div>
                <div>
                  <strong>{item.title}</strong>
                  <span>{item.detail}</span>
                </div>
                <span className={item.status === "done" ? "badge done" : "badge planned"}>
                  {item.status === "done" ? "Terminé" : "À venir"}
                </span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel model-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">Modèle actuel</span>
              <h2>{String(best.model).toUpperCase()} + hand-local</h2>
            </div>
            <Sparkles size={20} />
          </div>
          <div className="model-score">
            <strong>{(best.top1_mean * 100).toFixed(1)}</strong>
            <span>% Top-1</span>
          </div>
          <div className="score-grid">
            <div><span>Top-5</span><strong>{(best.top5_mean * 100).toFixed(1)}%</strong></div>
            <div><span>Macro-F1</span><strong>{best.macro_f1_mean.toFixed(3)}</strong></div>
            <div><span>Paramètres</span><strong>{Math.round(best.params).toLocaleString("fr-FR")}</strong></div>
            <div><span>Taille</span><strong>{best.model_mb.toFixed(2)} MB</strong></div>
          </div>
        </article>
      </section>

      <aside className="notice">
        <Info size={18} />
        <p>
          <strong>Limite actuelle.</strong> Le système reconnaît des mots isolés et ne remplace pas un interprète qualifié.
        </p>
      </aside>
    </>
  );
}

function Dataset() {
  const [data, setData] = useState(null);
  const [search, setSearch] = useState("");
  useEffect(() => { getJson(`/api/dataset?search=${encodeURIComponent(search)}&limit=200`).then(setData); }, [search]);
  if (!data) return <Loading />;

  return (
    <>
      <PageHeader eyebrow="Données" title="Exploration du dataset" description="Métadonnées WLASL, distribution des classes et benchmark." />
      <section className="metric-grid">
        <Metric label="Vidéos affichées" value={data.videos} />
        <Metric label="Signes" value={data.signs} />
        <Metric label="Signers" value={data.signers} />
        <Metric label="Durée moyenne" value={`${data.duration_mean} s`} />
      </section>

      <div className="search-box">
        <Search size={17} />
        <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Rechercher un signe..." />
      </div>

      <section className="two-column">
        <article className="panel">
          <div className="panel-heading"><div><span className="panel-kicker">Classes</span><h2>Distribution des signes</h2></div></div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={data.sign_counts.slice(0, 20)} margin={{ top: 10, right: 12, left: -20, bottom: 80 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="sign" angle={-55} textAnchor="end" interval={0} fontSize={11} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="videos" fill="#176b87" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><span className="panel-kicker">Performance</span><h2>Extraction parallèle</h2></div></div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={data.benchmark}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="workers" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="videos_per_sec" fill="#df8c55" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="panel-caption">De 0,098 à 0,626 vidéo/s avec 1 à 8 workers.</p>
        </article>
      </section>

      <article className="panel table-panel">
        <div className="panel-heading"><div><span className="panel-kicker">Métadonnées</span><h2>Échantillon des vidéos</h2></div></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Signe</th><th>Signer</th><th>Split</th><th>Durée</th><th>FPS</th><th>Résolution</th></tr></thead>
            <tbody>
              {data.rows.slice(0, 30).map((row, index) => (
                <tr key={index}>
                  <td><strong>{row.label}</strong></td><td>{row.signer_id}</td><td>{row.original_split}</td>
                  <td>{Number(row.duration_sec).toFixed(2)} s</td><td>{Number(row.fps).toFixed(2)}</td>
                  <td>{row.width} × {row.height}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </>
  );
}

function Experiments() {
  const [runs, setRuns] = useState([]);
  useEffect(() => { getJson("/api/experiments").then((data) => setRuns(data.runs)); }, []);

  const chartRuns = runs.map((run) => ({
    ...run,
    top1: Number(run.top1_mean) * 100,
  }));
  const best = chartRuns[0];

  return (
    <>
      <PageHeader eyebrow="MLflow" title="Comparaison des modèles" description="Résultats validés sur des signers non vus." />
      {best && (
        <section className="metric-grid">
          <Metric label="Meilleur Top-1" value={`${best.top1.toFixed(1)}%`} accent />
          <Metric label="Top-5" value={`${(best.top5_mean * 100).toFixed(1)}%`} />
          <Metric label="Macro-F1" value={best.macro_f1_mean.toFixed(3)} />
          <Metric label="Taille" value={`${best.model_mb.toFixed(2)} MB`} />
        </section>
      )}

      <section className="two-column">
        <article className="panel">
          <div className="panel-heading"><div><span className="panel-kicker">Précision</span><h2>Top-1 par expérience</h2></div></div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={chartRuns} margin={{ top: 10, right: 12, left: -15, bottom: 75 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="run" angle={-48} textAnchor="end" interval={0} fontSize={10} />
                <YAxis unit="%" />
                <Tooltip />
                <Bar dataKey="top1" radius={[5, 5, 0, 0]}>
                  {chartRuns.map((run, index) => <Cell key={run.run} fill={index === 0 ? "#176b87" : "#9bbac3"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading"><div><span className="panel-kicker">Compromis</span><h2>Taille et précision</h2></div></div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart margin={{ top: 20, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="model_mb" name="Taille" unit=" MB" />
                <YAxis type="number" dataKey="top1" name="Top-1" unit="%" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Scatter name="Modèles" data={chartRuns} fill="#df8c55" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <article className="panel table-panel">
        <div className="panel-heading"><div><span className="panel-kicker">Résultats</span><h2>Table complète</h2></div></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Expérience</th><th>Modèle</th><th>Features</th><th>Top-1</th><th>Top-5</th><th>Macro-F1</th><th>Paramètres</th><th>Taille</th></tr></thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run}>
                  <td><strong>{run.run}</strong></td><td>{run.model}</td><td>{run.features}</td>
                  <td>{(run.top1_mean * 100).toFixed(1)} ± {(run.top1_std * 100).toFixed(1)}%</td>
                  <td>{(run.top5_mean * 100).toFixed(1)}%</td>
                  <td>{run.macro_f1_mean.toFixed(3)}</td>
                  <td>{Math.round(run.params).toLocaleString("fr-FR")}</td><td>{run.model_mb.toFixed(2)} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </>
  );
}

function LiveDemo() {
  const [data, setData] = useState(null);
  useEffect(() => { getJson("/api/overview").then(setData); }, []);
  if (!data) return <Loading />;

  const ready = data.final_checkpoints.length > 0;
  return (
    <>
      <PageHeader eyebrow="Inférence" title="Démo live" description="Webcam, vidéo et transcription de signes isolés." />

      <article className={ready ? "live-card ready" : "live-card"}>
        <div className="live-preview">
          <div className="camera-frame">
            <div className="camera-grid" />
            <Video size={38} />
            <span>{ready ? "Flux prêt" : "Checkpoint final requis"}</span>
          </div>
        </div>
        <div className="live-controls">
          <span className={ready ? "status-pill ready" : "status-pill"}>{ready ? "Prêt" : "Non entraîné"}</span>
          <h2>{ready ? "Modèle final disponible" : "Application live en attente"}</h2>
          <p>
            {ready
              ? "Sélectionnez le checkpoint final et lancez le script de reconnaissance."
              : "Aucun fichier final.pt n'est présent. Entraînez le modèle Transformer hand-local avant l'inférence."}
          </p>
          <button className="primary-button" disabled={!ready}>
            <Play size={17} /> Ouvrir la webcam
          </button>
        </div>
      </article>

      <section className="two-column">
        <article className="panel">
          <div className="panel-heading"><div><span className="panel-kicker">Lancer</span><h2>Entraînement final</h2></div></div>
          <pre>python -m src.train_gru --final --model transformer --hand-local --run-name transformer_final</pre>
        </article>
        <article className="panel">
          <div className="panel-heading"><div><span className="panel-kicker">Tester</span><h2>Détection live</h2></div></div>
          <pre>python -m src.live_demo --checkpoint models/transformer_final/final.pt</pre>
        </article>
      </section>
    </>
  );
}

function Loading() {
  return <div className="loading"><div className="spinner" /><span>Chargement des données...</span></div>;
}

createRoot(document.getElementById("root")).render(<App />);
