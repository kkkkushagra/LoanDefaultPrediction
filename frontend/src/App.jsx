import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import {
  Activity, ArrowDown, ArrowLeft, ArrowRight, ArrowUp, BarChart3,
  Calculator, ChevronRight, CircleHelp, ClipboardList, FlaskConical, Gauge, Play,
} from 'lucide-react'
import { request } from './api'

const Portfolio = lazy(() => import('./Portfolio'))

const views = {
  score: {
    label: 'Score',
    title: 'Score an application',
    description: 'Estimate default probability and see which factors drive the score.',
    icon: Calculator,
  },
  simulate: {
    label: 'What-if',
    title: 'Compare a scenario',
    description: 'Change one or more inputs and compare the two model estimates.',
    icon: FlaskConical,
  },
  portfolio: {
    label: 'Portfolio',
    title: 'Portfolio performance',
    description: 'Review the distribution and observed outcomes across risk bands.',
    icon: BarChart3,
  },
  reviews: {
    label: 'Review history',
    title: 'Review history',
    description: 'Prototype decision log. This local demo has no authentication.',
    icon: ClipboardList,
  },
}

const loanFields = [
  { key: 'AMT_CREDIT', label: 'Credit amount', hint: 'Requested loan amount', example: '300000' },
  { key: 'AMT_INCOME_TOTAL', label: 'Annual income', hint: 'Reported total income', example: '100000' },
  { key: 'AMT_ANNUITY', label: 'Loan annuity', hint: 'Use the same basis as the source data', example: '25000' },
]

const scoreLoanFields = [
  loanFields[0],
  loanFields[2],
  { key: 'AMT_GOODS_PRICE', label: 'Purchase price', hint: 'Same monetary units as the other fields', example: '350000' },
  { key: 'NAME_CONTRACT_TYPE', label: 'Loan type', categoryKey: 'NAME_CONTRACT_TYPE', type: 'select' },
]
const incomeFields = [
  loanFields[1],
  { key: 'DAYS_EMPLOYED', label: 'Time in current employment', hint: 'Enter whole months; zero is valid', example: '120', type: 'months' },
  { key: 'NAME_INCOME_TYPE', label: 'Income type', categoryKey: 'NAME_INCOME_TYPE', type: 'select' },
]
const fields = [...scoreLoanFields, ...incomeFields]
const sample = {
  AMT_CREDIT: '300000',
  AMT_INCOME_TOTAL: '100000',
  AMT_ANNUITY: '25000',
  AMT_GOODS_PRICE: '350000',
  DAYS_EMPLOYED: '120',
  NAME_CONTRACT_TYPE: 'Cash loans',
  NAME_INCOME_TYPE: 'Working',
}
const percent = (value) => (value * 100).toFixed(2) + '%'

function toPayload(values, includedFields = fields) {
  return Object.fromEntries(includedFields
    .filter(({ key }) => values[key] !== undefined && values[key] !== '')
    .map(({ key, type }) => [
      key,
      key === 'DAYS_EMPLOYED'
        ? -Math.round(Number(values[key]) * 365.25 / 12)
        : type === 'select' ? values[key] : Number(values[key]),
    ]))
}

function useJob() {
  const [state, setState] = useState({ data: null, error: '', pending: false })
  const generation = useRef(0)

  function reset() {
    generation.current += 1
    setState({ data: null, error: '', pending: false })
  }

  async function run(task) {
    const current = ++generation.current
    setState({ data: null, error: '', pending: true })
    try {
      const data = await task()
      if (current === generation.current) {
        setState({ data, error: '', pending: false })
      }
    } catch (error) {
      if (current === generation.current) {
        setState({ data: null, error: error.message, pending: false })
      }
    }
  }

  return { ...state, reset, run }
}

function ErrorMessage({ error }) {
  return error && <p className="error" role="alert">{error}</p>
}

function ApplicationFields({ prefix, values, onChange, fields: inputFields, categories = {} }) {
  return inputFields.map(({ key, label, hint, example, type, categoryKey }) => (
    <div className="field" key={key}>
      <label htmlFor={prefix + key}>{label}</label>
      {type === 'select' ? (
        <select
          id={prefix + key}
          value={values[key] ?? ''}
          onChange={(event) => onChange({ ...values, [key]: event.target.value })}
          aria-describedby={prefix + key + '-hint'}
        >
          <option value="">Select if known</option>
          {(categories[categoryKey] ?? []).map((category) => (
            <option value={category} key={category}>{category}</option>
          ))}
        </select>
      ) : (
        <input
          id={prefix + key}
          type="number"
          min="0"
          step={type === 'months' ? '1' : 'any'}
          value={values[key] ?? ''}
          placeholder={prefix === 'scenario-' ? 'Keep original' : 'e.g. ' + example}
          onChange={(event) => onChange({ ...values, [key]: event.target.value })}
          aria-describedby={prefix + key + '-hint'}
        />
      )}
      <span className="field-hint" id={prefix + key + '-hint'}>{hint}</span>
    </div>
  ))
}

function ScoreSummary({ title, result }) {
  if (!result) {
    return (
      <section className="panel" aria-labelledby="risk-title">
        <div className="flex items-center justify-between gap-3">
          <h2 id="risk-title">Risk assessment</h2>
          <Gauge size={20} className="muted" aria-hidden="true" />
        </div>
        <div className="empty-score">
          <div className="empty-icon"><Activity size={24} aria-hidden="true" /></div>
          <p>Awaiting application score</p>
          <span>Submit the form to see the estimated default probability.</span>
        </div>
      </section>
    )
  }

  const {
    calibrated_probability: probability, risk_band: band, decision, cost_threshold: threshold,
    assumed_costs: assumedCosts,
  } = result

  return (
    <section className="panel">
      <h2>{title}</h2>
      <div className="score-number">
        <span>{percent(probability)}</span>
        <span className={'band ' + band.toLowerCase()}><span />{band} risk</span>
      </div>
      <p className="muted text-sm">Calibrated default probability</p>
      <div className="meter" aria-hidden="true"><span style={{ width: percent(probability) }} /></div>
      <div className="flex justify-between text-xs muted"><span>0%</span><span>100%</span></div>
      <div className="decision">
        <strong>{decision ? 'Flagged for review' : 'Below review threshold'}</strong>
        <span className="muted">{percent(threshold)} threshold</span>
      </div>
      <p className="text-xs muted">
        Validation-curve threshold · ASSUMED FN={assumedCosts.fn_cost}, FP={assumedCosts.fp_cost}
      </p>
    </section>
  )
}

function ReasonList({ reasons, scale }) {
  if (!reasons.length) return <p className="muted text-sm">No factors returned in this direction.</p>

  return (
    <ul className="reasons">
      {reasons.map(({ feature, shap_value, provided }, index) => (
        <li
          key={feature + index}
          className={provided === false ? 'reason-unprovided' : ''}
        >
          <div className="flex items-baseline justify-between gap-3">
            <span>{feature}</span>
            <span className={shap_value > 0 ? 'raising' : 'lowering'}>
              {shap_value > 0 ? '+' : ''}{shap_value.toFixed(3)}
            </span>
          </div>
          {provided === false && (
            <span className="reason-unprovided-label">
              history not provided (treated as missing)
            </span>
          )}
          <div className="reason-track" aria-hidden="true">
            <span
              className={shap_value > 0 ? 'positive' : 'negative'}
              style={{ width: percent(Math.abs(shap_value) / scale) }}
            />
          </div>
        </li>
      ))}
    </ul>
  )
}

function ScoreView({ values, update, categories, scoring, explanation, onSimulate }) {
  const applicantRef = useRef(null)
  const [reviewReason, setReviewReason] = useState('')
  const review = useJob()
  const applicant = toPayload(values)
  const filled = Object.keys(applicant).length
  const busy = scoring.pending || explanation.pending
  const high = (scoring.data?.top_reasons ?? []).filter(({ shap_value }) => shap_value > 0)
  const low = (explanation.data?.lowering_risk ?? []).filter(({ shap_value }) => shap_value < 0)
  const scale = Math.max(...[...high, ...low].map(({ shap_value }) => Math.abs(shap_value)), 0.0001)

  function submit(event) {
    event.preventDefault()
    if (!filled || busy) return
    explanation.reset()
    applicantRef.current = 'demo-' + crypto.randomUUID()
    scoring.run(() => request('/predict', applicant))
  }

  function submitReview(event) {
    event.preventDefault()
    const reason = reviewReason.trim()
    const officerDecision = event.nativeEvent.submitter?.value
    if (!reason || !officerDecision || !scoring.data || !applicantRef.current) return
    review.run(async () => {
      const saved = await request('/reviews', {
        applicant_ref: applicantRef.current,
        probability: scoring.data.calibrated_probability,
        risk_band: scoring.data.risk_band,
        officer_decision: officerDecision,
        reason_text: reason,
      })
      setReviewReason('')
      return saved
    })
  }

  return (
    <div className="score-layout">
      <section className="panel">
        <div className="section-header">
          <div><h2>Application details</h2><p>Use the same monetary units as the source data.</p></div>
          <button type="button" className="button secondary" disabled={busy} onClick={() => update(sample)}>
            Load example
          </button>
        </div>
        <form onSubmit={submit}>
          <fieldset className="input-group" disabled={busy}>
            <legend>Loan details</legend>
            <div className="fields score-fields">
              <ApplicationFields
                prefix="score-" values={values} onChange={update}
                fields={scoreLoanFields} categories={categories}
              />
            </div>
          </fieldset>
          <fieldset className="input-group" disabled={busy}>
            <legend>Income and employment</legend>
            <div className="fields score-fields">
              <ApplicationFields
                prefix="score-" values={values} onChange={update}
                fields={incomeFields} categories={categories}
              />
            </div>
          </fieldset>
          <div className="note">
            <CircleHelp size={17} aria-hidden="true" />
            <p>
              {filled} of {fields.length} application fields provided. Blank fields are omitted and treated as missing.
            </p>
          </div>
          <ErrorMessage error={scoring.error} />
          <button className="button primary" type="submit" disabled={!filled || busy}>
            <Play size={16} aria-hidden="true" />{scoring.pending ? 'Scoring…' : 'Score application'}
          </button>
        </form>
      </section>

      <div className="space-y-4" aria-live="polite">
        <ScoreSummary title="Risk assessment" result={scoring.data} />
        {scoring.data && (
          <section className="panel">
            <div className="section-header">
              <div>
                <h2>Score drivers</h2>
                <p>
                  SHAP contributions to the model score. For 7-field inputs, reasons may
                  reflect missing external history.
                </p>
              </div>
            </div>
            <h3 className="reason-heading"><ArrowUp size={15} aria-hidden="true" /> Raises score</h3>
            <ReasonList reasons={high} scale={scale} />
            {explanation.data ? (
              <>
                <h3 className="reason-heading"><ArrowDown size={15} aria-hidden="true" /> Lowers score</h3>
                <ReasonList reasons={low} scale={scale} />
              </>
            ) : (
              <button
                className="button secondary mt-4"
                disabled={explanation.pending}
                onClick={() => explanation.run(() => request('/explain', applicant))}
              >
                {explanation.pending ? 'Loading…' : 'Show factors that lower risk'}
              </button>
            )}
            <ErrorMessage error={explanation.error} />
            <div className="review-actions">
              <h3>Record officer decision <span className="prototype-label">Prototype · no auth</span></h3>
              <form onSubmit={submitReview}>
                <label className="field">
                  <span className="field-label">Reason (required)</span>
                  <textarea
                    required
                    maxLength={2000}
                    rows={3}
                    value={reviewReason}
                    onChange={(event) => setReviewReason(event.target.value)}
                    placeholder="Explain the decision"
                  />
                </label>
                <div className="review-buttons">
                  {['approve', 'refer', 'decline'].map((decision) => (
                    <button
                      className="button secondary"
                      disabled={review.pending || !reviewReason.trim()}
                      key={decision}
                      type="submit"
                      value={decision}
                    >
                      {decision[0].toUpperCase() + decision.slice(1)}
                    </button>
                  ))}
                </div>
              </form>
              {review.data && <p className="success" role="status">Decision saved to the prototype audit log.</p>}
              <ErrorMessage error={review.error} />
            </div>
            <button className="text-button mt-4" onClick={onSimulate}>
              Compare a scenario <ArrowRight size={16} aria-hidden="true" />
            </button>
          </section>
        )}
      </div>
    </div>
  )
}

function WhatIfView({ values, overrides, update, simulation, onScore }) {
  const applicant = toPayload(values)
  const changes = toPayload(overrides, loanFields)
  const hasBaseline = Object.keys(applicant).length > 0
  const canCompare = hasBaseline && Object.keys(changes).length > 0 && !simulation.pending
  const result = simulation.data
  const delta = result && (result.hypothetical.calibrated_probability - result.base.calibrated_probability) * 100
  const roundedDelta = delta === null ? null : Number(delta.toFixed(2))
  const deltaText = roundedDelta === null ? '' : new Intl.NumberFormat('en', {
    signDisplay: 'exceptZero',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(roundedDelta) + ' percentage points'

  function submit(event) {
    event.preventDefault()
    if (canCompare) {
      simulation.run(() => request('/simulate', { applicant, overrides: changes }))
    }
  }

  return (
    <div className="space-y-4">
      <section className="panel">
        <div className="section-header">
          <div><h2>Scenario inputs</h2><p>Leave a field blank to keep the current application value.</p></div>
          <button className="text-button" onClick={onScore}>
            <ArrowLeft size={16} aria-hidden="true" /> Edit application
          </button>
        </div>
        <dl className="baseline">
          {fields.map(({ key, label }) => (
            <div key={key}>
              <dt>{label}</dt>
              <dd>{values[key] === undefined || values[key] === ''
                ? 'Missing'
                : key === 'DAYS_EMPLOYED'
                  ? values[key] + ' months'
                  : applicant[key]?.toLocaleString('en', { maximumFractionDigits: 12 })}</dd>
            </div>
          ))}
        </dl>
        <form onSubmit={submit}>
          <fieldset className="fields scenario-fields" disabled={simulation.pending || !hasBaseline}>
            <ApplicationFields prefix="scenario-" values={overrides} onChange={update} fields={loanFields} />
          </fieldset>
          <div className="note">
            <CircleHelp size={17} aria-hidden="true" />
            <p>{hasBaseline
              ? 'This comparison shows how the model responds to hypothetical inputs. Other provided application details are carried through unchanged.'
              : 'Enter application details in Score to create a baseline.'}</p>
          </div>
          <ErrorMessage error={simulation.error} />
          <button className="button primary" type="submit" disabled={!canCompare}>
            <FlaskConical size={16} aria-hidden="true" />{simulation.pending ? 'Comparing…' : 'Run comparison'}
          </button>
        </form>
      </section>
      {result && (
        <div className="space-y-4" aria-live="polite">
          <p className="panel delta">
            <strong>Change in default probability</strong>
            <span>{roundedDelta === 0 ? 'No difference at the displayed precision' : deltaText}</span>
          </p>
          <div className="comparison">
            <ScoreSummary result={result.base} title="Original application" />
            <ScoreSummary result={result.hypothetical} title="Hypothetical application" />
          </div>
        </div>
      )}
    </div>
  )
}

function RemoteView({ path, children }) {
  const job = useJob()
  useEffect(() => { job.run(() => request(path)) }, [path])

  if (job.error) {
    return (
      <section className="panel">
        <ErrorMessage error={job.error} />
        <button className="button secondary" onClick={() => job.run(() => request(path))}>Try again</button>
      </section>
    )
  }

  return job.data ? children(job.data) : <p className="panel muted" role="status">Loading…</p>
}

function ReviewHistory({ data }) {
  return (
    <section className="panel space-y-4">
      <div>
        <h2>Review history <span className="prototype-label">Prototype · no authentication</span></h2>
        <p className="muted text-sm">Local SQLite audit log; decisions are not routed or access-controlled.</p>
      </div>
      {data.reviews.length === 0 ? (
        <p className="muted text-sm">No decisions have been recorded.</p>
      ) : (
        <ul className="review-history">
          {data.reviews.map((item) => (
            <li key={item.id}>
              <div className="review-history-heading">
                <strong>{item.officer_decision}</strong>
                <span>{new Date(item.created_at).toLocaleString()}</span>
              </div>
              <p>{item.reason_text}</p>
              <span className="muted text-xs">
                {item.applicant_ref} · {item.risk_band} · {percent(item.probability)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export function App() {
  const [view, setView] = useState('score')
  const [values, setValues] = useState({})
  const [overrides, setOverrides] = useState({})
  const [categories, setCategories] = useState({})
  const [connected, setConnected] = useState(null)
  const scoring = useJob()
  const explanation = useJob()
  const simulation = useJob()
  const current = views[view]

  async function checkApi() {
    setConnected(null)
    try {
      const [health, options] = await Promise.all([
        request('/health'),
        request('/application-options'),
      ])
      if (health.status !== 'ok') throw new Error('API is unhealthy')
      if (!options.categories?.NAME_CONTRACT_TYPE || !options.categories?.NAME_INCOME_TYPE) {
        throw new Error('API application options are unavailable')
      }
      setCategories(options.categories)
    } catch {
      setConnected(false)
      return
    }
    setConnected(true)
  }

  useEffect(() => { checkApi() }, [])

  function updateApplication(next) {
    setValues(next)
    setOverrides({})
    scoring.reset()
    explanation.reset()
    simulation.reset()
  }

  function updateOverrides(next) {
    setOverrides(next)
    simulation.reset()
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={22} aria-hidden="true" /></div>
          <div><strong>Loan Risk</strong><span>Officer dashboard</span></div>
        </div>
        <nav aria-label="Dashboard views" className="navigation">
          {Object.entries(views).map(([key, item]) => (
            <button key={key} aria-current={view === key ? 'page' : undefined} onClick={() => setView(key)}>
              <item.icon size={18} aria-hidden="true" /><span>{item.label}</span>
              {view === key && <ChevronRight size={15} className="nav-arrow" aria-hidden="true" />}
            </button>
          ))}
        </nav>
        <button
          className="api-status" type="button" onClick={checkApi}
          title="Recheck API connection" disabled={connected === null}
        >
          <span className={connected === null ? 'checking' : connected ? 'online' : 'offline'} />
          {connected === null ? 'Checking API…' : connected ? 'API connected' : 'API offline · Retry'}
        </button>
      </aside>

      <main className="content">
        <header className="page-header"><h1>{current.title}</h1><p>{current.description}</p></header>
        {view === 'score' && (
          <ScoreView
            values={values} update={updateApplication} categories={categories}
            scoring={scoring} explanation={explanation} onSimulate={() => setView('simulate')}
          />
        )}
        {view === 'simulate' && (
          <WhatIfView
            values={values} overrides={overrides} update={updateOverrides}
            simulation={simulation} onScore={() => setView('score')}
          />
        )}
        {view === 'portfolio' && (
          <RemoteView path="/portfolio/summary">
            {(data) => (
              <Suspense fallback={<p className="panel muted" role="status">Loading chart…</p>}>
                <Portfolio rows={data.rows} />
              </Suspense>
            )}
          </RemoteView>
        )}
        {view === 'reviews' && (
          <RemoteView path="/reviews">{(data) => <ReviewHistory data={data} />}</RemoteView>
        )}
      </main>
    </div>
  )
}