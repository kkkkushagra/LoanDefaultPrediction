import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const count = (value) => value.toLocaleString('en')
const percentage = (value) => (value * 100).toFixed(2) + '%'
const axis = { fill: '#a8afc2', fontSize: 12 }

export default function Portfolio({ rows }) {
  const total = rows.reduce((sum, row) => sum + row.count, 0)

  return (
    <div className="space-y-4">
      <section className="panel" aria-labelledby="band-chart">
        <div className="section-header">
          <div>
            <h2 id="band-chart">Risk band distribution</h2>
            <p>{count(total)} applicants in the portfolio summary.</p>
          </div>
        </div>
        <div className="portfolio-chart" aria-hidden="true">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 12, right: 12, bottom: 5, left: 0 }} barGap={6}>
              <CartesianGrid vertical={false} stroke="#303649" strokeDasharray="4 4" />
              <XAxis dataKey="band" tick={axis} tickLine={false} axisLine={false} />
              <YAxis
                width={45}
                tick={axis}
                tickFormatter={(value) => Math.round(value * 100) + '%'}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{ background: '#181d2a', border: '1px solid #3b455d', borderRadius: 8 }}
                itemStyle={{ color: '#eef1f7' }}
                cursor={{ fill: '#ffffff09' }}
                formatter={percentage}
              />
              <Legend />
              <Bar
                dataKey="applicant_share" name="Applicant share" fill="#a59bff"
                maxBarSize={42} radius={[5, 5, 0, 0]} isAnimationActive={false}
              />
              <Bar
                dataKey="observed_default_rate" name="Observed default rate" fill="#66d8bd"
                maxBarSize={42} radius={[5, 5, 0, 0]} isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
      <section className="panel">
        <div className="section-header"><h2>Risk band breakdown</h2></div>
        <div className="table-scroll">
          <table aria-label="Risk band breakdown">
            <thead>
              <tr>
                <th scope="col">Risk band</th>
                <th scope="col">Applicants</th>
                <th scope="col">Applicant share</th>
                <th scope="col">Observed default rate</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.band}>
                  <th scope="row">
                    <span className={'band ' + row.band.toLowerCase()}><span />{row.band}</span>
                  </th>
                  <td>{count(row.count)}</td>
                  <td>{percentage(row.applicant_share)}</td>
                  <td>{percentage(row.observed_default_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}