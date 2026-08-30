import { useState, useEffect } from 'react'

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [orgId, setOrgId] = useState('Org A (Telecommunications)');
  const [query, setQuery] = useState('Analyze our core business health and find anomalies.');
  const [status, setStatus] = useState('idle'); // idle, loading, success, error
  const [results, setResults] = useState(null);
  const [activeStep, setActiveStep] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');

  const fetchModels = async () => {
    try {
      setModels(['Loading from backend...']);
      const res = await fetch('http://localhost:8000/api/models');
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Backend error or API Key missing in .env");
      setModels(data.models);
      if (data.models.length > 0) setSelectedModel(data.models[0]);
    } catch (err) {
      alert(err.message);
      setModels([]);
    }
  };

  const runPipeline = async () => {
    if (!selectedModel) {
      alert("Please fetch/select a model first.");
      return;
    }
    
    setStatus('loading');
    setResults(null);
    setErrorMsg('');
    setActiveStep(1);

    // Flowchart Animation Loop
    const stepInterval = setInterval(() => {
      setActiveStep(prev => (prev < 4 ? prev + 1 : 4));
    }, 1200);

    try {
      const res = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, org_id: orgId, model: selectedModel })
      });
      
      clearInterval(stepInterval);
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail || "Server Error");
      
      setResults(data);
      setStatus('success');
    } catch (err) {
      clearInterval(stepInterval);
      setErrorMsg(err.message);
      setStatus('error');
    }
  };

  const statusTexts = [
    "Initializing...",
    "Running Vector Search (TF-IDF)...",
    "Prompting LLM for Schema Mapping...",
    "Executing Scikit-Learn Anomaly Detection...",
    "Synthesizing Executive Narrative..."
  ];

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="glass-panel sidebar">
        <h2>⚙️ Control Panel</h2>
        
        <button onClick={fetchModels} className="glow-btn">
          Connect Backend & Fetch Models
        </button>

        <div className="input-group">
          <label>🤖 Select LLM Model</label>
          <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
            {models.length === 0 && <option value="">Fetch models first...</option>}
            {models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label>🏢 Select Organization</label>
          <select value={orgId} onChange={e => setOrgId(e.target.value)}>
            <option value="Org A (Telecommunications)">Org A (Telecommunications)</option>
            <option value="Org B (Supply Chain)">Org B (Supply Chain)</option>
          </select>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header>
          <h1>📈 BusinessIntelligence.ai</h1>
          <p className="subtitle">Enterprise AI Analyst</p>
        </header>

        <div className="glass-panel query-section">
          <input 
            type="text" 
            value={query} 
            onChange={e => setQuery(e.target.value)} 
          />
          <button onClick={runPipeline} className="glow-btn run-btn" disabled={status === 'loading'}>
            {status === 'loading' ? 'Running...' : 'Run Pipeline'}
          </button>
        </div>

        {status === 'loading' && (
          <div className="loader">
            <div className="flowchart">
              {[1, 2, 3, 4].map(step => (
                <div key={step} style={{ display: 'contents' }}>
                  <div className={`flow-step ${activeStep === step ? 'active-step' : ''}`}>
                    {step}. {
                      step === 1 ? '🔍 Vector RAG Retrieval' :
                      step === 2 ? '🤖 LLM Schema Mapper' :
                      step === 3 ? '⚙️ ML Quantitative Engine' :
                      '📄 Executive Synthesis'
                    }
                  </div>
                  {step < 4 && <div className="flow-arrow">⬇</div>}
                </div>
              ))}
            </div>
            <div className="spinner"></div>
            <p>{statusTexts[activeStep]}</p>
          </div>
        )}

        {status === 'error' && (
          <div className="glass-panel" style={{ borderColor: 'red' }}>
            <h3 style={{ color: 'red' }}>Pipeline Error</h3>
            <p>{errorMsg}</p>
          </div>
        )}

        {status === 'success' && results && (
          <div className="results-grid">
            <div className="glass-panel evidence-panel">
              <h3>⚙️ Execution Logs</h3>
              <div className="logs">
                <p>✅ Data fetched (<b>{results.rows_analyzed}</b> rows).</p>
                <p>🎯 Inferred KPI: <b>{results.mapped_kpi}</b></p>
                <p>🔍 Discovered Drivers: <b>{results.mapped_drivers.join(', ')}</b></p>
              </div>
              
              <details className="rag-details">
                <summary>Show RAG Retrieved Business Context</summary>
                <pre>{JSON.stringify(results.retrieved_rag_context, null, 2)}</pre>
              </details>
              
              <details>
                <summary>Show Evidence Graph (JSON)</summary>
                <pre>{JSON.stringify(results.evidence_graph, null, 2)}</pre>
              </details>
            </div>

            <div className="glass-panel narrative-panel">
              <h3>📄 Action Narrative</h3>
              <div 
                className="story-content" 
                dangerouslySetInnerHTML={{ 
                  __html: results.story.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>') 
                }} 
              />
              <div className="telemetry">
                <span>Latency: {results.telemetry.latency_ms} ms</span> | 
                <span> Tokens: {results.telemetry.tokens || 'N/A'}</span> | 
                <span> Cost: ${results.telemetry.cost_usd.toFixed(5)}</span>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
