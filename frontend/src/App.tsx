import { useCallback, useEffect, useState } from "react";
import {
  analyzeImage,
  clearToken,
  deleteScan,
  fetchMe,
  getScan,
  listScans,
  login,
  register,
  setToken,
  type PredictionResponse,
  type ScanDetail,
  type ScanSummary,
  type User,
} from "./api";
import "./App.css";

type Tab = "classify" | "history" | "about";

function App() {
  const [tab, setTab] = useState<Tab>("classify");
  const [user, setUser] = useState<User | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register" | null>(null);
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [showGradcam, setShowGradcam] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const [scans, setScans] = useState<ScanSummary[]>([]);
  const [selectedScan, setSelectedScan] = useState<ScanDetail | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    fetchMe()
      .then(setUser)
      .catch(() => {
        clearToken();
        setUser(null);
      });
  }, []);

  const loadHistory = useCallback(async () => {
    if (!user) return;
    setHistoryLoading(true);
    try {
      setScans(await listScans());
    } catch {
      setScans([]);
    } finally {
      setHistoryLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (tab === "history" && user) loadHistory();
  }, [tab, user, loadHistory]);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeImage(file, showGradcam);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const onAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setAuthError(null);
    try {
      const res =
        authMode === "register"
          ? await register(authEmail, authPassword)
          : await login(authEmail, authPassword);
      setToken(res.access_token);
      setUser(res.user);
      setAuthMode(null);
      setAuthEmail("");
      setAuthPassword("");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const onLogout = () => {
    clearToken();
    setUser(null);
    setScans([]);
    setSelectedScan(null);
    if (tab === "history") setTab("classify");
  };

  const onViewScan = async (id: number) => {
    try {
      setSelectedScan(await getScan(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load scan");
    }
  };

  const onDeleteScan = async (id: number) => {
    try {
      await deleteScan(id);
      setScans((prev) => prev.filter((s) => s.id !== id));
      if (selectedScan?.id === id) setSelectedScan(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🧠</span>
            <div>
              <h1>NeuroScan</h1>
              <p>Brain MRI tumor classification with Grad-CAM explainability</p>
            </div>
          </div>

          <div className="header-right">
            <nav className="tabs">
              <button
                className={tab === "classify" ? "tab active" : "tab"}
                onClick={() => setTab("classify")}
              >
                Classify MRI
              </button>
              {user && (
                <button
                  className={tab === "history" ? "tab active" : "tab"}
                  onClick={() => setTab("history")}
                >
                  History
                </button>
              )}
              <button
                className={tab === "about" ? "tab active" : "tab"}
                onClick={() => setTab("about")}
              >
                About
              </button>
            </nav>

            <div className="auth-area">
              {user ? (
                <>
                  <span className="user-email">{user.email}</span>
                  <button className="btn-secondary" onClick={onLogout}>
                    Log out
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="btn-secondary"
                    onClick={() => {
                      setAuthMode("login");
                      setAuthError(null);
                    }}
                  >
                    Log in
                  </button>
                  <button
                    className="btn-primary btn-sm"
                    onClick={() => {
                      setAuthMode("register");
                      setAuthError(null);
                    }}
                  >
                    Sign up
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {authMode && (
        <div className="modal-overlay" onClick={() => setAuthMode(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>{authMode === "login" ? "Log in" : "Create account"}</h2>
            <form onSubmit={onAuth}>
              <label>
                Email
                <input
                  type="email"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  required
                />
              </label>
              <label>
                Password
                <input
                  type="password"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  minLength={8}
                  required
                />
              </label>
              {authError && <div className="banner error">{authError}</div>}
              <button className="btn-primary" disabled={authLoading}>
                {authLoading
                  ? "Please wait…"
                  : authMode === "login"
                    ? "Log in"
                    : "Sign up"}
              </button>
            </form>
          </div>
        </div>
      )}

      <main className="main">
        {tab === "classify" && (
          <div className="classify-grid">
            <section className="panel upload-panel">
              <h2>Upload MRI Scan</h2>
              {!user && (
                <p className="hint-banner">
                  Log in to save scans to your history automatically.
                </p>
              )}

              <div
                className={`dropzone ${dragOver ? "drag-over" : ""} ${preview ? "has-image" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => document.getElementById("file-input")?.click()}
              >
                {preview ? (
                  <img src={preview} alt="MRI preview" className="preview-img" />
                ) : (
                  <div className="dropzone-placeholder">
                    <span className="upload-icon">📁</span>
                    <p>Drag & drop a brain MRI image</p>
                    <span className="hint">PNG, JPG, or JPEG</span>
                  </div>
                )}
                <input
                  id="file-input"
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFile(f);
                  }}
                />
              </div>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={showGradcam}
                  onChange={(e) => setShowGradcam(e.target.checked)}
                />
                Show Grad-CAM heatmap
              </label>

              <button
                className="btn-primary"
                disabled={!file || loading}
                onClick={onAnalyze}
              >
                {loading ? "Analyzing…" : "Analyze"}
              </button>
            </section>

            <section className="panel result-panel">
              <h2>Results</h2>

              {!result && !error && !loading && (
                <div className="empty-state">
                  <span>🔬</span>
                  <p>Upload an MRI scan and click Analyze to see predictions</p>
                </div>
              )}

              {loading && (
                <div className="loading-state">
                  <div className="spinner" />
                  <p>Running inference…</p>
                </div>
              )}

              {error && <div className="banner error">{error}</div>}

              {result && (
                <div className="results">
                  {result.scan_id && (
                    <div className="banner success">
                      Saved to your scan history (#{result.scan_id})
                    </div>
                  )}
                  <div className="prediction-card">
                    <span className="prediction-label">Prediction</span>
                    <h3>{result.prediction_label}</h3>
                    <div className="confidence">
                      <span className="confidence-value">
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                      <span className="confidence-text">confidence</span>
                    </div>
                  </div>

                  <div className="probabilities">
                    <h4>Class Probabilities</h4>
                    {sortedProbs.map(([label, prob]) => (
                      <div key={label} className="prob-row">
                        <div className="prob-header">
                          <span>{label}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="prob-bar">
                          <div
                            className="prob-fill"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {result.gradcam_image && (
                    <div className="gradcam-section">
                      <h4>Grad-CAM — Model Attention</h4>
                      <p className="gradcam-caption">
                        Regions influencing the prediction
                      </p>
                      <img
                        src={`data:image/png;base64,${result.gradcam_image}`}
                        alt="Grad-CAM heatmap overlay"
                        className="gradcam-img"
                      />
                    </div>
                  )}
                </div>
              )}
            </section>
          </div>
        )}

        {tab === "history" && user && (
          <div className="history-grid">
            <section className="panel">
              <h2>Scan History</h2>
              {historyLoading && <p className="hint">Loading…</p>}
              {!historyLoading && scans.length === 0 && (
                <div className="empty-state">
                  <p>No saved scans yet. Analyze an MRI while logged in.</p>
                </div>
              )}
              <ul className="scan-list">
                {scans.map((scan) => (
                  <li
                    key={scan.id}
                    className={
                      selectedScan?.id === scan.id
                        ? "scan-item active"
                        : "scan-item"
                    }
                  >
                    <button
                      className="scan-item-btn"
                      onClick={() => onViewScan(scan.id)}
                    >
                      <strong>{scan.prediction_label}</strong>
                      <span>{scan.filename}</span>
                      <span>
                        {(scan.confidence * 100).toFixed(1)}% ·{" "}
                        {new Date(scan.created_at).toLocaleString()}
                      </span>
                    </button>
                    <button
                      className="btn-danger-sm"
                      onClick={() => onDeleteScan(scan.id)}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            </section>

            <section className="panel">
              <h2>Scan Detail</h2>
              {!selectedScan && (
                <div className="empty-state">
                  <p>Select a scan from the list to view details</p>
                </div>
              )}
              {selectedScan && (
                <div className="results">
                  <div className="prediction-card">
                    <span className="prediction-label">Prediction</span>
                    <h3>{selectedScan.prediction_label}</h3>
                    <div className="confidence">
                      <span className="confidence-value">
                        {(selectedScan.confidence * 100).toFixed(1)}%
                      </span>
                      <span className="confidence-text">confidence</span>
                    </div>
                  </div>
                  {selectedScan.image_thumbnail && (
                    <img
                      src={`data:image/jpeg;base64,${selectedScan.image_thumbnail}`}
                      alt="Scan thumbnail"
                      className="gradcam-img"
                    />
                  )}
                  {selectedScan.gradcam_image && (
                    <img
                      src={`data:image/png;base64,${selectedScan.gradcam_image}`}
                      alt="Grad-CAM"
                      className="gradcam-img"
                    />
                  )}
                </div>
              )}
            </section>
          </div>
        )}

        {tab === "about" && (
          <section className="panel about-panel">
            <h2>About NeuroScan</h2>
            <p>
              NeuroScan classifies brain MRI scans into four categories: glioma,
              meningioma, pituitary tumor, or no tumor.
            </p>

            <h3>Deep Learning Stack</h3>
            <ul>
              <li>
                Fine-tuned <strong>EfficientNet-B0</strong> (or ResNet-18) with
                transfer learning
              </li>
              <li>Data augmentation, early stopping, learning-rate scheduling</li>
              <li>
                <strong>Grad-CAM</strong> saliency maps for interpretability
              </li>
            </ul>

            <h3>Architecture</h3>
            <ul>
              <li>
                <strong>Frontend:</strong> React + TypeScript (Vite) on Vercel
              </li>
              <li>
                <strong>Backend:</strong> FastAPI + PyTorch on Render
              </li>
              <li>
                <strong>Database:</strong> PostgreSQL (users + scan history)
              </li>
            </ul>

            <h3>Dataset</h3>
            <p>
              <a
                href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
                target="_blank"
                rel="noopener noreferrer"
              >
                Brain Tumor MRI (Kaggle)
              </a>
            </p>

            <blockquote className="disclaimer">
              This is a research/education demo — not for clinical diagnosis.
            </blockquote>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>NeuroScan · Brain MRI Tumor Classification</p>
      </footer>
    </div>
  );
}

export default App;
