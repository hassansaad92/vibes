-- Run once in Supabase → SQL Editor (or psql)

CREATE TABLE IF NOT EXISTS predictions (
    id                  uuid             DEFAULT gen_random_uuid() PRIMARY KEY,
    machine_id          text             NOT NULL,
    filename            text             NOT NULL,
    features            jsonb            NOT NULL,
    predicted_label     text             NOT NULL CHECK (predicted_label IN ('f', 'h')),
    probability_faulty  double precision NOT NULL,
    probability_healthy double precision NOT NULL,
    created_at          timestamptz      DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_machine_created
    ON predictions (machine_id, created_at DESC);
