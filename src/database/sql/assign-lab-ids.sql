  DROP MATERIALIZED VIEW IF EXISTS assign_lab_ids CASCADE;
CREATE MATERIALIZED VIEW assign_lab_ids AS

        -- create a table which has fuzzy boundaries on ICU admission
        -- involves first creating a lag/lead version of intime/outtime
  WITH i AS
       (
        SELECT subject_id, hadm_id, icustay_id, intime, outtime
            -- find the outtime of the previous icustay and the intime of the next icustay
               , LAG (outtime) OVER (partition BY subject_id, hadm_id ORDER BY intime) AS outtime_lag
               , LEAD (intime) OVER (partition BY subject_id, hadm_id ORDER BY intime) AS intime_lead
          FROM icustays
        )

       , iid_assign AS
       (
        SELECT i.subject_id, i.hadm_id, i.icustay_id
               , admittime, dischtime,  intime, outtime, outtime_lag, intime_lead
            -- for the first icustay, everything starting from hospital admission to outtime 
           -- is assigned to this icustay id. For subsequent icustays, only data from the 
           -- end of the previous icustay to the end of the currect icustay gets assigned to these ids. 
           -- For the last icustay in the sequence, then everything from icu intime to the
           -- end of hospital admission is included
            -- in order for the data extraction to be as inclusive as possible almost all 
           -- lab events are included using the following boundaries (3 days before and 
           -- after discharge - actually only 2.5 days / 60 hours before is required but
           -- this is easier for the purposed of having a hospital admission boundary). 
               , CASE WHEN i.outtime_lag IS NOT NULL 
                 THEN i.outtime_lag
                 ELSE ad.admittime - INTERVAL '72' HOUR
                  END AS data_start
               , CASE WHEN i.intime_lead IS NOT NULL 
                 THEN i.outtime
                 ELSE ad.dischtime + INTERVAL '72' HOUR
                  END AS data_end
          FROM i
         INNER JOIN admissions ad ON i.hadm_id = ad.hadm_id
        )
        -- also create fuzzy boundaries on hospitalization
        -- Note that sometimes the discharge time of the previous hospital occurs after the  next 
        -- hospital admission causing a negative difference, in that case the boundary here is 
        -- set before the discharge time of the previous stay/after admittime of next hospital stay
        -- No duplicate data appears to be created if we drop this requirement and this is 
        -- likely to how the labevents table was created in the first place.
       , h AS
       (
        SELECT subject_id, hadm_id, admittime, dischtime
               , LAG (dischtime) OVER (PARTITION BY subject_id ORDER BY admittime) AS dischtime_lag
               , LEAD (admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS admittime_lead
        FROM admissions
        )
        
       , adm AS
       (
        SELECT h.subject_id, h.hadm_id
           , admittime, dischtime
            -- this rule is:
            --  if there are two hospitalizations within 24 hours, set the start/stop
            --  time as half way between the two admissions
               , CASE WHEN h.dischtime_lag IS NOT NULL
                       AND h.dischtime_lag > (h.admittime - INTERVAL '144' HOUR)
                 THEN h.admittime - ((h.admittime - h.dischtime_lag)/2)
                 ELSE h.admittime - INTERVAL '72' HOUR
                 END AS data_start
               , CASE WHEN h.admittime_lead IS NOT NULL
                       AND h.admittime_lead < (h.dischtime + INTERVAL '144' HOUR)
                 THEN h.dischtime + ((h.admittime_lead - h.dischtime)/2)
                 ELSE (h.dischtime + INTERVAL '72' HOUR)
                 END AS data_end
          FROM h
        )
        

SELECT   adm.subject_id, adm.hadm_id, id.icustay_id, id.data_start AS id_start
       , id.data_end AS id_end, adm.data_start, adm.data_end
       , GREATEST(id.data_start, adm.data_start) AS starttime
       , LEAST(id.data_end, adm.data_end) AS endtime
  FROM adm 
  LEFT JOIN iid_assign id
    ON id.hadm_id = adm.hadm_id