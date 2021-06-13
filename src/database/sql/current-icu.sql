/* We create a list of all observed times for our chart and lab measurement, 
then we use the transfer table to find out their precise location */

  DROP MATERIALIZED VIEW IF EXISTS current_icu CASCADE;
CREATE MATERIALIZED VIEW current_icu AS

SELECT times.subject_id, times.icustay_id, times.charttime, curr_careunit
  FROM
       (SELECT c.subject_id, c.icustay_id, c.charttime 
          FROM extracted_chart c
         UNION
        SELECT lab.subject_id, lab.icustay_id, lab.charttime 
          FROM extracted_labs lab
        ) times
  LEFT JOIN transfers t
    ON times.subject_id = t.subject_id
   AND times.charttime >= t.intime AND times.charttime < t.outtime
 ORDER BY times.subject_id, times.charttime