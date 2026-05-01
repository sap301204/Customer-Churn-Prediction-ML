export default function Home() {
  const rows = [
    { id: "CUST-1021", p: 0.82, action: "Retention call + discount" },
    { id: "CUST-1099", p: 0.71, action: "Support callback" },
    { id: "CUST-1172", p: 0.58, action: "Engagement campaign" }
  ];

  return (
    <main style={{ maxWidth: 1000, margin: "40px auto", fontFamily: "Arial" }}>
      <h1>Customer Churn Success-Ops Dashboard</h1>
      <p>Demo Next.js dashboard. For deployment, Streamlit dashboard is easier.</p>
      <table width="100%" cellPadding="12">
        <thead>
          <tr><th>Customer</th><th>Churn Probability</th><th>Action</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td><td>{(r.p * 100).toFixed(1)}%</td><td>{r.action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
