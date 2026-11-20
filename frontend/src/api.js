export async function request(path, payload) {
  const response = await fetch('/api' + path, {
    signal: AbortSignal.timeout(30_000),
    ...(payload === undefined ? {} : {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  }).catch((error) => {
    const message = error.name === 'TimeoutError'
      ? 'The API took too long to respond. Try again.'
      : 'Could not reach the scoring API. Check that FastAPI is running.'
    throw new Error(message)
  })

  if (!response.ok) {
    if (response.status === 422 || response.status === 400) {
      throw new Error('The API rejected the application. Check the input values and try again.')
    }
    throw new Error('The scoring API returned an error (' + response.status + ').')
  }

  return response.json().catch(() => {
    throw new Error('The API response was incomplete. Check the proxy target and try again.')
  })
}