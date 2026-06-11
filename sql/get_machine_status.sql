SELECT predicted_label
FROM predictions
WHERE machine_id = %(machine_id)s
ORDER BY created_at DESC
LIMIT 10;
