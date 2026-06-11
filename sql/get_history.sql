SELECT id, machine_id, filename, predicted_label, probability_faulty, probability_healthy, created_at
FROM predictions
WHERE machine_id = %(machine_id)s
ORDER BY created_at DESC
LIMIT %(limit)s;
