export const EVENT_LOG_LIMIT = 1000;

export function appendBoundedEvents<T>(current: T[], additions: T[], limit = EVENT_LOG_LIMIT) {
  if (limit < 1) return [];
  return [...current, ...additions].slice(-limit);
}
