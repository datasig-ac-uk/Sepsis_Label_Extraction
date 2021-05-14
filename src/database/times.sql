DROP MATERIALIZED VIEW IF EXISTS times CASCADE;
CREATE MATERIALIZED VIEW times AS
-- create a table which has fuzzy boundaries on hospital admission
-- involves first creating a lag/lead version of disch/admit time
WITH i AS
(
  SELECT
    subject_id, hadm_id, icustay_id, intime, outtime
    -- find the outtime of the previous icustay and the intime of the next icustay
    , LAG (outtime) OVER (PARTITION BY subject_id, hadm_id ORDER BY intime) AS outtime_lag
    , LEAD (intime) OVER (PARTITION BY subject_id, hadm_id ORDER BY intime) AS intime_lead
  FROM icustays
)

, iid_assign AS
(
  SELECT
    i.subject_id, i.hadm_id, i.icustay_id
    , admittime, dischtime,  intime, outtime, outtime_lag, intime_lead -- this line is just for checking purposes, can comment out
    -- for the first icustay, then everything starting from hospital admission to outtime is assigned to this icustay id
    -- for subsequent icustays, only data from the end of the previous icustay to the end of the currect icustay gets assigned to these ids. For the last icustay in the sequence, then everything from icu intime to the end of hospital admission is included
    -- in order for the data extraction to be as inclusive as possible almost all lab events are included using the following boundaries (3 days before and after discharge - actually only 2.5 days / 60 hours before is required but this is easier for the purposed of having a hospital admission boundary).
    , CASE
        WHEN i.outtime_lag IS NOT NULL THEN i.outtime_lag
      ELSE ad.admittime - INTERVAL '72' HOUR
      END AS data_start
    , CASE
        WHEN i.intime_lead IS NOT NULL THEN i.outtime
      ELSE ad.dischtime + INTERVAL '72' HOUR
      END AS data_end
    FROM i
    INNER JOIN admissions ad ON i.hadm_id = ad.hadm_id
)
-- also create fuzzy boundaries on hospitalization
-- Note that sometimes the discharge time of the previous hospital occurs after the next hospital admission causing a negative difference, in that case the boundary here is set before the discharge time of the previous stay/after admittime of next hospital stay
-- No duplicate data appears to be created if we drop this requirement and this is likely to how the labevents table was created in the first place.
, h AS
(
  SELECT
    subject_id, hadm_id, admittime, dischtime
    , LAG (dischtime) OVER (PARTITION BY subject_id ORDER BY admittime) AS dischtime_lag
    , LEAD (admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS admittime_lead
  FROM admissions
)
, adm AS
(
  SELECT
    h.subject_id, h.hadm_id
    -- this rule is:
    --  if there are two hospitalizations within 24 hours, set the start/stop
    --  time as half way between the two admissions
    , CASE
        WHEN h.dischtime_lag IS NOT NULL
        AND h.dischtime_lag > (h.admittime - INTERVAL '144' HOUR)
          THEN h.admittime - ((h.admittime - h.dischtime_lag)/2)
      ELSE h.admittime - INTERVAL '72' HOUR
      END AS data_start
    , CASE
        WHEN h.admittime_lead IS NOT NULL
        AND h.admittime_lead < (h.dischtime + INTERVAL '144' HOUR)
          THEN h.dischtime + ((h.admittime_lead - h.dischtime)/2)
      ELSE (h.dischtime + INTERVAL '72' HOUR)
      END AS data_end
    FROM h
)
, times AS 
(
SELECT id.subject_id, id.hadm_id, id.icustay_id, id.data_start AS id_start, id.data_end AS id_end, adm.data_start, adm.data_end, GREATEST(id.data_start, adm.data_start) AS starttime_hr
, LEAST(id.data_end, adm.data_end) AS endtime_hr
FROM iid_assign id
INNER JOIN adm
ON id.hadm_id = adm.hadm_id
)
    
, all_hours AS
(
  SELECT
    t.hadm_id,
    t.icustay_id

    , DATE_TRUNC('hour', t.starttime_hr ) AS starttime

    -- create integers for each charttime in hours from admission
    -- so 0 is admission time, 1 is one hour after admission, etc, up to ICU disch
    , GENERATE_SERIES
    (
      0,
      CEIL(EXTRACT(EPOCH FROM t.endtime_hr-t.starttime_hr)/60.0/60.0)::INTEGER
    ) AS hr

  FROM times t
)

SELECT
  ah.hadm_id
  , ah.icustay_id
  , ah.hr
  -- add the hr series
  -- endtime now indexes the end time of every hour for each patient
  , ah.hr*(INTERVAL '1' HOUR) + starttime AS starttime
  , ah.hr*(INTERVAL '1' HOUR) + starttime +  INTERVAL '1' HOUR AS endtime
FROM all_hours ah
ORDER BY ah.icustay_id, hr;
