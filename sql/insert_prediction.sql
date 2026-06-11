INSERT INTO predictions
    (machine_id, filename, features, predicted_label, probability_faulty, probability_healthy)
VALUES
    (%(machine_id)s, %(filename)s, %(features)s, %(predicted_label)s, %(probability_faulty)s, %(probability_healthy)s)
RETURNING id, created_at;
