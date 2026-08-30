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
  const [feedback, setFeedback] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [jwtToken, setJwtToken] = useState('');
  const [role, setRole] = useState('');
  
  const [databases, setDatabases] = useState([]);
  const [showAddDb, setShowAddDb] = useState(false);
  const [newDb, setNewDb] = useState({ org_name: '', connection_string: '', query: '' });

  useEffect(() => {
    fetchDatabases();
  }, []);

  const fetchDatabases = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/databases');
      const data = await res.json();
      setDatabases(data.databases || []);
      if (data.databases && data.databases.length > 0 && !orgId) {
        setOrgId(data.databases[0]);
      }
    } catch (err) {
      console.error("Error fetching databases:", err);
    }
  };

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

  const loginPersona = async (selectedRole) => {
    setRole(selectedRole);
    if (!selectedRole) {
      setJwtToken('');
      return;
    }
    try {
      const res = await fetch('http://localhost:8000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: selectedRole })
      });
      const data = await res.json();
      if (res.ok) {
        setJwtToken(data.access_token);
      } else {
        alert(data.detail || "Login failed");
      }
    } catch (err) {
      alert("Error logging in: " + err.message);
    }
  };

  const addDatabase = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('http://localhost:8000/api/databases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newDb)
      });
      const data = await res.json();
      if (res.ok) {
        alert(data.message);
        setShowAddDb(false);
        setNewDb({ org_name: '', connection_string: '', query: '' });
        fetchDatabases();
      } else {
        alert("Failed to add database: " + data.detail);
      }
    } catch (err) {
      alert("Error: " + err.message);
    }
  };

  const runPipeline = async () => {
    if (!selectedModel) {
      alert("Please fetch/select a model first.");
      return;
    }
    if (!jwtToken) {
      alert("Please select a Persona (Login) first.");
      return;
    }
    
    setStatus('loading');
    setResults(null);
    setErrorMsg('');
    setFeedback('');
    setFeedbackStatus('');
    setActiveStep(1);

    // Flowchart Animation Loop
    const stepInterval = setInterval(() => {
      setActiveStep(prev => (prev < 4 ? prev + 1 : 4));
    }, 1200);

    try {
      const res = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${jwtToken}`
        },
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

  const submitFeedback = async () => {
    if (!feedback) return;
    try {
      setFeedbackStatus('Submitting...');
      const res = await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kpi: results.mapped_kpi, correction: feedback, user_id: 'user_123' })
      });
      const data = await res.json();
      if (res.ok) {
        setFeedbackStatus('✅ Feedback Loop Active! Model logic tuning queued.');
        setFeedback('');
      } else {
        setFeedbackStatus('❌ Error submitting feedback.');
      }
    } catch (err) {
      setFeedbackStatus('❌ Error submitting feedback.');
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
        
        <div className="input-group" style={{ marginBottom: '20px', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
          <label>👤 Authenticate Persona</label>
          <select value={role} onChange={e => loginPersona(e.target.value)}>
            <option value="">-- Select Persona to Login --</option>
            <option value="cmo">Chief Marketing Officer (Full Access)</option>
            <option value="analyst">Data Analyst (Restricted Financials)</option>
          </select>
          {jwtToken && <small style={{ color: '#4ade80' }}>✓ JWT Token Secured</small>}
        </div>

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
          <div style={{ display: 'flex', gap: '10px' }}>
            <select value={orgId} onChange={e => setOrgId(e.target.value)} style={{ flex: 1 }}>
              {databases.map(db => (
                <option key={db} value={db}>{db}</option>
              ))}
            </select>
            <button onClick={() => setShowAddDb(!showAddDb)} className="icon-btn" title="Add New Database">
              ➕
            </button>
          </div>
        </div>

        {showAddDb && (
          <form onSubmit={addDatabase} className="input-group add-db-form" style={{ padding: '15px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', marginBottom: '20px' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#4ade80' }}>Add Database</h4>
            <input 
              type="text" 
              placeholder="Organization Name" 
              required
              value={newDb.org_name}
              onChange={e => setNewDb({...newDb, org_name: e.target.value})}
              style={{ marginBottom: '10px', width: '100%' }}
            />
            <input 
              type="text" 
              placeholder="Connection String (e.g. postgresql://...)" 
              required
              value={newDb.connection_string}
              onChange={e => setNewDb({...newDb, connection_string: e.target.value})}
              style={{ marginBottom: '10px', width: '100%' }}
            />
            <textarea 
              placeholder="SQL Query to fetch data" 
              required
              value={newDb.query}
              onChange={e => setNewDb({...newDb, query: e.target.value})}
              style={{ marginBottom: '10px', width: '100%', height: '80px', resize: 'vertical' }}
            />
            <button type="submit" className="glow-btn" style={{ padding: '8px' }}>Save Database</button>
          </form>
        )}
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
              <div className="story-content">
                {results.story.status === "Complete" ? (
                  <>
                    <p><strong>🚨 Finding:</strong> {results.story.change_detection}</p>
                    <p><strong>🧠 Contributing Factors:</strong> {results.story.contributing_factors}</p>
                    <p><strong>🎯 Action Plan:</strong> {results.story.recommended_actions}</p>
                  </>
                ) : (
                  <p>{results.story.change_detection}</p>
                )}
              </div>
              <div className="telemetry">
                <span>Latency: {results.telemetry.latency_ms} ms</span> | 
                <span> Tokens: {results.telemetry.tokens || 'N/A'}</span> | 
                <span> Cost: ${results.telemetry.cost_usd.toFixed(5)}</span>
              </div>
              
              <div className="feedback-section" style={{ marginTop: '20px', padding: '15px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                <h4>🔄 Human Feedback Loop</h4>
                <p style={{ fontSize: '0.85em', color: '#aaa', marginBottom: '10px' }}>Notice a hallucination or incorrect root cause? Correct the AI to tune its Knowledge Base.</p>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <input 
                    type="text" 
                    value={feedback} 
                    onChange={e => setFeedback(e.target.value)} 
                    placeholder="e.g., Actually, this churn spike was due to a billing error..."
                    style={{ flex: 1 }}
                  />
                  <button onClick={submitFeedback} className="glow-btn">Submit Correction</button>
                </div>
                {feedbackStatus && <p style={{ fontSize: '0.9em', color: '#4ade80', marginTop: '10px' }}>{feedbackStatus}</p>}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
