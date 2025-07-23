import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { Bar, Pie, Line } from 'react-chartjs-2';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  LineElement,
  PointElement
} from 'chart.js';
import { Tooltip as ReactTooltip } from 'react-tooltip';
import { FaCalendarAlt, FaArrowUp, FaArrowDown, FaInfoCircle } from 'react-icons/fa';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, LineElement, PointElement);

function App() {
  const [account, setAccount] = useState('ACC001');
  const [accountSummary, setAccountSummary] = useState(null);
  const [clusterData, setClusterData] = useState(null);
  const [clusterError, setClusterError] = useState('');
  const [endDate, setEndDate] = useState('');
  const [spendPrediction, setSpendPrediction] = useState(null);
  const [spendError, setSpendError] = useState('');
  const [monthEndPrediction, setMonthEndPrediction] = useState(null);
  const [monthEndError, setMonthEndError] = useState('');
  const [spendCategories, setSpendCategories] = useState(null);
  const [costSuggestions, setCostSuggestions] = useState(null);
  const [costError, setCostError] = useState('');
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState('');
  const [startDate, setStartDate] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState('');
  const [forecastMonths, setForecastMonths] = useState(3);
  const [spendingForecast, setSpendingForecast] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState('');
  const [fraudData, setFraudData] = useState({ flagged: [], all: [] });
  const [fraudLoading, setFraudLoading] = useState(false);
  const [fraudError, setFraudError] = useState('');

  const API_URL = 'http://localhost:8000';

  // Fetch account summary, cluster, cost, health on analyze
  const analyzeAccount = async () => {
    setClusterError('');
    setClusterData(null);
    setCostError('');
    setCostSuggestions(null);
    setHealthError('');
    setHealth(null);
    try {
      const [clusterRes, costRes, healthRes] = await Promise.all([
        axios.get(`${API_URL}/cluster/${account}`),
        axios.get(`${API_URL}/cost_cut_suggestions/${account}`),
        axios.get(`${API_URL}/financial_health/${account}`)
      ]);
      if (clusterRes.data.error) setClusterError(clusterRes.data.error);
      else setClusterData(clusterRes.data);
      if (costRes.data.error) setCostError(costRes.data.error);
      else setCostSuggestions(costRes.data);
      if (healthRes.data.advice) setHealth(healthRes.data);
      else setHealthError('No health data');
    } catch (err) {
      setClusterError('Error fetching cluster');
      setCostError('Error fetching suggestions');
      setHealthError('Error fetching health');
    }
  };

  // Predict spend for custom range
  const predictSpend = async () => {
    setSpendError('');
    setSpendPrediction(null);
    if (!startDate || !endDate) {
      setSpendError('Select both start and end date');
      return;
    }
    try {
      const res = await axios.post(`${API_URL}/predict_spend/${account}`, {
        start_date: startDate.toISOString().slice(0, 10),
        end_date: endDate.toISOString().slice(0, 10),
        mode: 'custom'
      });
      if (res.data.error) setSpendError(res.data.error);
      else setSpendPrediction(res.data);
    } catch (err) {
      setSpendError('Error fetching prediction');
    }
  };

  // Predict till month end
  const predictTillMonthEnd = async () => {
    setMonthEndError('');
    setMonthEndPrediction(null);
    try {
      const res = await axios.post(`${API_URL}/predict_spend/${account}`, {
        mode: 'month_end'
      });
      if (res.data.error) setMonthEndError(res.data.error);
      else setMonthEndPrediction(res.data);
    } catch (err) {
      setMonthEndError('Error fetching month-end prediction');
    }
  };

  // Gemini chat handler
  const sendChat = async () => {
    if (!chatInput.trim()) return;
    setChatError('');
    setChatLoading(true);
    setChatHistory(h => [...h, { role: 'user', text: chatInput }]);
    try {
      const res = await axios.post('http://localhost:8000/gemini_chat', {
        question: chatInput,
        account_no: account
      });
      if (res.data.error) setChatError(res.data.error);
      else setChatHistory(h => [...h, { role: 'assistant', text: res.data.answer }]);
    } catch (e) {
      setChatError('Error contacting Gemini assistant.');
    }
    setChatInput('');
    setChatLoading(false);
  };

  // Fetch future forecast on account change or analyze
  const fetchSpendingForecast = async (acct = account, months = forecastMonths) => {
    setForecastLoading(true);
    setForecastError('');
    setSpendingForecast(null);
    try {
      const resp = await axios.post(`${API_URL}/spending_forecast`, { account_no: acct, forecast_months: months });
      const data = resp.data;
      if (data.error) throw new Error(data.error);
      setSpendingForecast(data);
    } catch (e) {
      setForecastError(e.message || 'Error fetching forecast');
    } finally {
      setForecastLoading(false);
    }
  };
  // Fetch on account/Analyze/forecastMonths change
  useEffect(() => {
    if (account) fetchSpendingForecast(account, forecastMonths);
    // eslint-disable-next-line
  }, [account, forecastMonths]);

  // Fetch fraud data on account change or analyze
  const fetchFraudData = async (acct = account) => {
    setFraudLoading(true);
    setFraudError('');
    setFraudData({ flagged: [], all: [] });
    try {
      const res = await axios.get(`${API_URL}/detect_fraud/${acct}`);
      if (res.data.error) setFraudError(res.data.error);
      else setFraudData(res.data);
    } catch (err) {
      setFraudError('Error fetching fraud data');
    }
    setFraudLoading(false);
  };
  useEffect(() => {
    if (account) fetchFraudData(account);
    // eslint-disable-next-line
  }, [account]);

  // Render helpers
  const renderCategoryChart = (data, title) => {
    if (!data || Object.keys(data).length === 0) return <div>No data</div>;
    return (
      <Pie
        data={{
          labels: Object.keys(data),
          datasets: [
            {
              data: Object.values(data),
              backgroundColor: [
                '#6366f1', '#f59e42', '#10b981', '#f43f5e', '#3b82f6', '#fbbf24', '#a21caf', '#14b8a6', '#eab308', '#ef4444'
              ]
            }
          ]
        }}
        options={{ plugins: { legend: { position: 'bottom' }, title: { display: true, text: title } } }}
      />
    );
  };

  return (
    <div className="container">
      <h1>Cashflow Analysis Dashboard</h1>
      <div className="dashboard-grid">
        {/* Account/Cluster Analysis */}
        <div className="card">
          <h2>Account Analysis</h2>
          <div className="input-group">
            <input type="text" value={account} onChange={e => setAccount(e.target.value)} placeholder="Account Number" />
            <button onClick={analyzeAccount}>Analyze Account</button>
          </div>
          {clusterError && <div className="error-box">{clusterError}</div>}
          {clusterData && (
            <>
              <div><b>Account:</b> {account}</div>
              <div><b>Cluster:</b> {clusterData.cluster} ({clusterData.type})</div>
              <div><b>Description:</b> {clusterData.description}</div>
              <div><b>Average Amount:</b> ${clusterData.avg_amount?.toFixed(2)}</div>
              <div><b>Total Spend:</b> ${clusterData.total_spend?.toFixed(2)}</div>
              <div><b>Number of Transactions:</b> {clusterData.num_transactions}</div>
              <div><b>Average Transaction:</b> ${clusterData.avg_transaction?.toFixed(2)}</div>
              <div><b>Most Frequent Category:</b> {clusterData.most_frequent_category || 'N/A'}</div>
              <div><b>Last Transaction Date:</b> {clusterData.last_transaction_date || 'N/A'}</div>
              <div style={{marginTop: 12}}>
                <b>Monthly Trend (last 6 months):</b>
                {clusterData.monthly_trend && Object.keys(clusterData.monthly_trend).length > 0 ? (
                  <Bar
                    data={{
                      labels: Object.keys(clusterData.monthly_trend),
                      datasets: [{
                        label: 'Monthly Spend',
                        data: Object.values(clusterData.monthly_trend),
                        backgroundColor: '#6366f1'
                      }]
                    }}
                    options={{ plugins: { legend: { display: false } } }}
                  />
                ) : <div style={{color:'#888'}}>No trend data</div>}
              </div>
              <div style={{marginTop: 12}}>
                <b>Cluster Center:</b> {clusterData.cluster_center && clusterData.cluster_center.length > 0 ? clusterData.cluster_center.map((v, i) => <span key={i}>{v.toFixed(2)}{i === 0 ? ' (Avg Amount), ' : ' (Total Spend)'}</span>) : 'N/A'}
              </div>
              <div style={{marginTop: 8}}>
                <b>All Cluster Centers:</b>
                {clusterData.all_centers && clusterData.all_centers.length > 0 ? (
                  <ul style={{margin:0,paddingLeft:18}}>
                    {clusterData.all_centers.map((c, idx) => (
                      <li key={idx}>Cluster {idx}: {c.map((v, i) => `${v.toFixed(2)}${i === 0 ? ' (Avg Amount)' : ' (Total Spend)'}`).join(', ')}</li>
                    ))}
                  </ul>
                ) : 'N/A'}
              </div>
            </>
          )}
        </div>

        {/* Financial Health */}
        <div className="card">
          <h2>Financial Health</h2>
          {healthError && <div className="error-box">{healthError}</div>}
          {health && (
            <>
              <div className="health-score">Score: <b>{health.score}</b> / 100</div>
              <div>Percentile: <b>{health.percentile?.toFixed(1)}</b></div>
              <ul>
                {health.advice?.map((a, i) => <li key={i}>{a}</li>)}
              </ul>
            </>
          )}
        </div>

        {/* Fraud Detection */}
        <div className="card fraud-card">
          <h2>Fraud Detection</h2>
          {fraudLoading ? <div className="loading">Loading fraud analysis...</div> : null}
          {fraudError ? <div className="error-box">{fraudError}</div> : null}
          {fraudData.flagged && fraudData.flagged.length > 0 ? (
            <>
              <div className="fraud-table-wrap">
                <table className="fraud-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Amount</th>
                      <th>Category</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fraudData.flagged.map((txn, i) => (
                      <tr key={i}>
                        <td>{txn.date}</td>
                        <td>${txn.amount?.toFixed(2)}</td>
                        <td>{txn.txn_category}</td>
                        <td>{txn.fraud_score?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="fraud-chart-wrap">
                <Bar
                  data={{
                    labels: fraudData.all.map(txn => txn.date),
                    datasets: [{
                      label: 'Fraud Score',
                      data: fraudData.all.map(txn => txn.fraud_score),
                      backgroundColor: fraudData.all.map(txn => txn.is_fraud ? '#f43f5e' : '#6366f1'),
                      borderRadius: 4
                    }]
                  }}
                  options={{
                    plugins: { legend: { display: false }, title: { display: true, text: 'Fraud Score Over Time' } },
                    responsive: true,
                    scales: { x: { title: { display: true, text: 'Date' } }, y: { title: { display: true, text: 'Score' } } }
                  }}
                />
              </div>
            </>
          ) : !fraudLoading && !fraudError ? <div className="no-fraud-msg">No suspicious transactions detected.</div> : null}
        </div>

        {/* Cost-Cutting Suggestions */}
        <div className="card">
          <h2>Cost-Cutting Suggestions</h2>
          {costError && <div className="error-box">{costError}</div>}
          {costSuggestions && (
            <>
              <ul>
                {costSuggestions.suggestions?.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
              <div className="simulated-savings">
                <b>Simulated Savings:</b>
                <ul>
                  {Object.entries(costSuggestions.simulated_savings || {}).map(([cat, val]) => (
                    <li key={cat}>{cat}: ${val.toFixed(2)}</li>
                  ))}
                </ul>
              </div>
              <div className="rising-cats">
                <b>Rising Categories:</b> {costSuggestions.rising_categories?.join(', ') || 'None'}
              </div>
              <div className="trend">
                <b>Trends (% change last 30d):</b>
                <ul>
                  {Object.entries(costSuggestions.trend || {}).map(([cat, val]) => (
                    <li key={cat}>{cat}: {val === null ? 'N/A' : `${val}%`}</li>
                  ))}
                </ul>
              </div>
              {costSuggestions.gemini_explanation && (
                <div className="explanation-box">{costSuggestions.gemini_explanation}</div>
              )}
            </>
          )}
        </div>

        {/* Spending Prediction */}
        <div className="card prediction-card">
          <h2>Spending Prediction</h2>
          <div className="prediction-input-row modern">
            <label><FaCalendarAlt /> Start
              <DatePicker
                selected={startDate}
                onChange={date => setStartDate(date)}
                placeholderText="Start Date"
                dateFormat="yyyy-MM-dd"
              />
            </label>
            <label><FaCalendarAlt /> End
              <DatePicker
                selected={endDate}
                onChange={date => setEndDate(date)}
                placeholderText="End Date"
                dateFormat="yyyy-MM-dd"
              />
            </label>
            <button className="primary-btn" onClick={predictSpend}>Predict</button>
            <button className="secondary-btn" onClick={predictTillMonthEnd}>Till Month End</button>
          </div>
          {spendError && <div className="error-box modern-error">{spendError}</div>}
          {((spendPrediction && !spendPrediction.error) || (monthEndPrediction && !monthEndPrediction.error)) ? (
            <div className="prediction-results modern">
              {spendPrediction && !spendPrediction.error && spendPrediction.predicted_total > 0 ? (
                <PredictionDetails data={spendPrediction} title="Custom Range" />
              ) : spendPrediction && !spendPrediction.error && spendPrediction.predicted_total === 0 ? (
                <div className="no-prediction-msg modern-no-data">No data available for prediction for this account/date range.</div>
              ) : null}
              {monthEndPrediction && !monthEndPrediction.error && monthEndPrediction.predicted_total > 0 ? (
                <PredictionDetails data={monthEndPrediction} title="Till Month End" />
              ) : monthEndPrediction && !monthEndPrediction.error && monthEndPrediction.predicted_total === 0 ? (
                <div className="no-prediction-msg modern-no-data">No data available for prediction for this account/date range.</div>
              ) : null}
            </div>
          ) : null}
        </div>
        {/* Spending Forecast */}
        <div className="card forecast-card">
          <h2>Spending Forecast</h2>
          <div className="forecast-controls">
            <label>Forecast Period:
              <select value={forecastMonths} onChange={e => setForecastMonths(Number(e.target.value))}>
                {[1,2,3,4,5,6].map(m => <option key={m} value={m}>{m} month{m>1?'s':''}</option>)}
              </select>
            </label>
          </div>
          {forecastLoading ? <div className="loading">Loading forecast...</div> : null}
          {forecastError ? <div className="error-box">{forecastError}</div> : null}
          {spendingForecast && spendingForecast.forecast ? (
            <>
              <div className="forecast-table-wrap">
                <table className="forecast-table">
                  <thead>
                    <tr>
                      <th>Month</th>
                      {spendingForecast.categories.map(cat => <th key={cat}>{cat}</th>)}
                      <th>Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {spendingForecast.forecast.map(row => (
                      <tr key={row.month}>
                        <td>{row.month}</td>
                        {spendingForecast.categories.map(cat => <td key={cat}>{row[cat] ? row[cat].toFixed(2) : '0.00'}</td>)}
                        <td>{row.total ? row.total.toFixed(2) : '0.00'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="forecast-chart-wrap">
                <Bar
                  data={{
                    labels: spendingForecast.forecast.map(row => row.month),
                    datasets: spendingForecast.categories.map((cat, i) => ({
                      label: cat,
                      data: spendingForecast.forecast.map(row => row[cat] || 0),
                      backgroundColor: `hsl(${i*50}, 70%, 60%)`,
                      stack: 'spend',
                    }))
                  }}
                  options={{
                    plugins: { legend: { position: 'top' } },
                    responsive: true,
                    scales: { x: { stacked: true }, y: { stacked: true } }
                  }}
                />
              </div>
            </>
          ) : null}
        </div>
      </div>
      {/* Smart Assistant Chat Widget */}
      <div className="card chat-widget">
        <h2>Smart Assistant</h2>
        <div className="chat-history">
          {chatHistory.length === 0 && <div className="chat-empty">Ask me anything about your finances!</div>}
          {chatHistory.map((msg, i) => (
            <div key={i} className={`chat-msg ${msg.role}`}>{msg.role === 'user' ? 'You: ' : 'Assistant: '}{msg.text}</div>
          ))}
          {chatLoading && <div className="chat-msg assistant">Assistant: <span className="chat-loading">...</span></div>}
        </div>
        <div className="chat-input-row">
          <input
            type="text"
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            placeholder="Type your question..."
            onKeyDown={e => { if (e.key === 'Enter') sendChat(); }}
            disabled={chatLoading}
          />
          <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()}>Send</button>
        </div>
        {chatError && <div className="error-box">{chatError}</div>}
      </div>
    </div>
  );
}

function PredictionDetails({ data, title }) {
  const colorList = [
    '#6366f1', '#f59e42', '#10b981', '#f43f5e', '#3b82f6', '#fbbf24', '#a21caf', '#14b8a6', '#eab308', '#ef4444'
  ];
  const catKeys = Object.keys(data.predicted_by_category || {});
  const histKeys = Object.keys(data.historical_by_category || {});
  const allCats = Array.from(new Set([...catKeys, ...histKeys]));
  // Calculate % difference
  const diff = data.historical_total > 0 ? ((data.predicted_total - data.historical_total) / data.historical_total) * 100 : null;
  const badge = diff !== null ? (
    <span className={`badge ${diff > 0 ? 'over' : 'under'}`}>{diff > 0 ? <FaArrowUp /> : <FaArrowDown />} {Math.abs(diff).toFixed(1)}% {diff > 0 ? 'Over' : 'Under'} Last Year</span>
  ) : null;
  // Prepare horizontal bar chart data
  const barData = {
    labels: catKeys,
    datasets: [
      {
        label: 'Predicted',
        data: catKeys.map(cat => data.predicted_by_category[cat] || 0),
        backgroundColor: colorList.slice(0, catKeys.length),
        borderRadius: 6,
        barThickness: 18
      }
    ]
  };
  // Prepare trend chart for top 3 categories (if trend data available)
  let topCats = catKeys.slice(0, 3);
  let trendChart = null;
  if (data.trend && topCats.length > 0) {
    trendChart = (
      <div className="trend-chart">
        <Line data={{
          labels: ["Prev 30d", "Last 30d"],
          datasets: topCats.map((cat, i) => ({
            label: cat,
            data: [0, data.trend[cat] || 0],
            borderColor: colorList[i % colorList.length],
            backgroundColor: colorList[i % colorList.length] + '33',
            tension: 0.4,
            fill: false
          }))
        }} options={{
          plugins: { legend: { position: 'bottom' }, title: { display: true, text: 'Trend for Top Categories' } },
          responsive: true,
          scales: { y: { beginAtZero: true, ticks: { callback: v => v + '%' } } }
        }} />
      </div>
    );
  }
  return (
    <div className="prediction-redesign">
      <div className="pred-left">
        <div className="pred-title-row">
          <div className="prediction-title">{title}</div>
          {badge}
        </div>
        <div className="pred-summary-row">
          <div className="predicted-total"><span>Predicted</span><b>${data.predicted_total?.toFixed(2)}</b></div>
          <div className="historical-total"><span>Last Year</span><b>${data.historical_total?.toFixed(2)}</b></div>
        </div>
        <div className="pred-bar-chart">
          {catKeys.length === 0 ? <div className="breakdown-empty">No category breakdown</div> : (
            <Bar data={barData} options={{
              indexAxis: 'y',
              plugins: { legend: { display: false }, title: { display: true, text: 'Predicted by Category' } },
              responsive: true,
              scales: { x: { beginAtZero: true } }
            }} />
          )}
        </div>
      </div>
      <div className="pred-right">
        {trendChart}
        {data.gemini_explanation && (
          <div className="ai-insight-callout"><FaInfoCircle className="ai-insight-icon" /> <span><b>AI Insight:</b> {data.gemini_explanation}</span></div>
        )}
        {data.trend && Object.keys(data.trend).length > 0 && (
          <div className="trend-breakdown">
            <b>Trend (% change last 30d vs previous 30d):</b>
            <ul>
              {Object.entries(data.trend).map(([cat, val], i) => (
                <li key={cat}>
                  <span className="cat-label">{cat}:</span> {val === null ? 'N/A' : `${val > 0 ? '+' : ''}${val}%`}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
