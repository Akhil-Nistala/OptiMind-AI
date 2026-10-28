CREATE TABLE IF NOT EXISTS production (
 production_date TEXT NOT NULL, job_id TEXT NOT NULL, machine_id TEXT NOT NULL,
 planned_minutes REAL NOT NULL CHECK(planned_minutes >= 0),
 actual_minutes REAL NOT NULL CHECK(actual_minutes >= 0),
 completed_quantity INTEGER NOT NULL CHECK(completed_quantity >= 0),
 downtime_minutes REAL CHECK(downtime_minutes >= 0),
 production_target INTEGER NOT NULL CHECK(production_target > 0),
 scheduled_minutes REAL NOT NULL CHECK(scheduled_minutes > 0),
 PRIMARY KEY(production_date,machine_id)
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
