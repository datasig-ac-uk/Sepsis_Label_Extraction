/* 
This is a variation on the main script to extract sepsis time, sepsis-time-blood.sql. Here we keep all cultures (which is what was included in the original sepsis-3 paper).

This sensitivity analysis is only done for the SOFA window 24 hours before and 12 hours after the time of suspected infection

Note this query does not do much in the way of finding suspicion of infections before ICU admission, in that case the prescriptions table should be incorporated

We explore the idea of splitting medications by different treatment cycles and removing patients with isolated antibiotic events 

To increase specificity of the method, we only take the samples that are blood cultures

We no longer filter out the patients who had an isolated antibiotic case

Some of the tables written by Alistair Johnson, we have changed the tables for exclusions by using inputevents table as we want a more accurate time

*/

  DROP MATERIALIZED VIEW IF EXISTS sepsis_cohort_time_sensitivity_2412 CASCADE;
CREATE MATERIALIZED VIEW sepsis_cohort_time_sensitivity_2412 AS


        -- Define the services that the patient is receiving care under
  WITH serv AS
       (
        SELECT hadm_id, curr_service
               , DENSE_RANK() OVER (PARTITION BY hadm_id ORDER BY transfertime) AS rn
          FROM services
        )

        -- select the antibiotics
       , abx_pres AS
       (
        SELECT pr.hadm_id
               , pr.drug AS antibiotic_name
               , pr.startdate AS antibiotic_time
               , pr.enddate AS antibiotic_endtime
               , pr.route
          FROM prescriptions pr
              -- inner join to subselect to only antibiotic prescriptions
         INNER JOIN abx_poe_list ab
            ON pr.drug = ab.drug
        )
        
        -- get cultures for each icustay
        -- note this duplicates prescriptions
        -- each ICU stay in the same hospitalization will get a copy of all prescriptions for that hospitalization
        -- we introduced as ICU stay sequence as well as whether the precription date was before icu admission date. We select only metavision data.
       , ab_tbl_pres AS
       (
        SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
               , ie.intime, ie.outtime
               , DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq
               , abx.antibiotic_name
               , abx.antibiotic_time
               , abx.antibiotic_endtime
               , abx.route
               -- Check if this patient has been given antibiotics before ICu admission 
               , CASE WHEN abx.antibiotic_time < CAST(ie.intime AS DATE) THEN 1 ELSE 0 END AS before_icu 
          FROM icustays ie
          LEFT JOIN admissions ad
            ON ie.hadm_id = ad.hadm_id
         INNER JOIN serv s
            ON ie.hadm_id = s.hadm_id 
           AND rn = 1 
           AND curr_service NOT IN ('CSURG','TSURG')
          LEFT JOIN abx_pres abx
            ON ie.hadm_id = abx.hadm_id 
         WHERE admission_type != 'ELECTIVE' 
           AND ie.dbsource = 'metavision'
        )
        
        -- the following query has all the icustay_ids of interest
       , filtered_ids AS
       (
        SELECT * 
          FROM (SELECT icustay_id, MAX(before_icu) AS before_icu
                  FROM ab_tbl_pres   
                 WHERE icustay_seq = 1 
                 GROUP BY hadm_id, icustay_id    
                 ORDER BY hadm_id, icustay_id) a
         WHERE a.before_icu = 0
        )

        -- Split the antibiotics into different cycles
        -- As mentioned in the sepsis-3 paper, isolated cases of antibiotics would not satisfy the requirement for a suspicion of infection. We need another dose of antibiotics within 96 hours of the first one
       , abx_partition AS
       ( SELECT *
                , SUM( new_antibiotics_cycle )
                  OVER ( partition BY icustay_id, antibiotic_name ORDER BY icustay_id, starttime )
             AS abx_num
           FROM
                (SELECT subject_id, hadm_id, icustay_id, starttime, endtime
                        , label AS antibiotic_name
                        , CASE WHEN starttime <= LAG(endtime) OVER 
                                                 (PARTITION BY icustay_id, label 
                                                      ORDER BY icustay_id, starttime) 
                                                 + INTERVAL '48' HOUR THEN 0 ELSE 1 END 
                                                         AS new_antibiotics_cycle
                        , LEAD(starttime) OVER (PARTITION BY icustay_id 
                                                    ORDER BY icustay_id, starttime) - endtime 
                                                       AS next_antibio
                        , CASE WHEN LEAD(starttime) OVER 
                              (PARTITION BY icustay_id ORDER BY icustay_id, starttime) - endtime 
                                     <= INTERVAL '96' HOUR THEN 0 ELSE 1 END AS isolated_case
                   FROM inputevents_mv input
                  INNER JOIN d_items d ON input.itemid = d.itemid
                  WHERE d.category ilike '%antibiotics%'
                  ORDER BY subject_id, hadm_id, icustay_id, starttime) A
                    -- order by subject_id, hadm_id, icustay_id, starttime
            )

        -- group the antibiotic information together to form courses of antibiotics and also to check whether they are isolated cases
        -- note the last drug dose taken by the patient will always be classed as an isolated case above. However, if this is of the same type as antibiotics given to patient within last 2 days, then it will be grouped together with other doses.
       , abx_partition_grouped AS
       (
        SELECT subject_id, hadm_id, icustay_id, MIN(starttime) AS starttime
               , MAX(endtime) AS endtime
               , COUNT(*) AS doses, antibiotic_name, MIN(isolated_case) AS isolated_case
          FROM abx_partition
         GROUP BY subject_id, hadm_id, icustay_id, antibiotic_name, abx_num
         ORDER BY subject_id, hadm_id, icustay_id, starttime
        )

        -- Set the antibiotic time and the start time if it is not an isolated case
       , ab_tbl AS
       (
        SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
               , ie.intime, ie.outtime
               , CASE WHEN ab.isolated_case = 0 THEN ab.antibiotic_name ELSE NULL END 
                    AS antibiotic_name
               , CASE WHEN ab.isolated_case = 0 THEN ab.starttime ELSE NULL END 
                    AS antibiotic_time
              --, ab.isolated_case
              --, abx.endtime
          FROM icustays ie
         INNER JOIN filtered_ids fil ON fil.icustay_id = ie.icustay_id
          LEFT JOIN abx_partition_grouped ab
            ON ie.hadm_id = ab.hadm_id 
           AND ie.icustay_id = ab.icustay_id
        )
        
        -- Find the microbiology events
       , me AS
       (
        SELECT hadm_id, chartdate, charttime
               , spec_type_desc
               , MAX(CASE WHEN org_name IS NOT NULL AND org_name != '' THEN 1 ELSE 0 END) AS PositiveCulture
          FROM microbiologyevents
         GROUP BY hadm_id, chartdate, charttime, spec_type_desc
        )
        
       , ab_fnl AS
       (
        SELECT ab_tbl.icustay_id, ab_tbl.intime, ab_tbl.outtime
               , ab_tbl.antibiotic_name
               , ab_tbl.antibiotic_time
               , coalesce(me.charttime,me.chartdate) AS culture_charttime
               , me.positiveculture AS positiveculture
               , me.spec_type_desc AS specimen
               , CASE WHEN COALESCE(antibiotic_time, COALESCE(me.charttime,me.chartdate)) IS NULL
                 THEN 0
                 ELSE 1 END AS suspected_infection
               , LEAST(antibiotic_time, COALESCE(me.charttime, me.chartdate)) AS t_suspicion
          FROM ab_tbl
          LEFT JOIN me 
            ON ab_tbl.hadm_id = me.hadm_id
           AND ab_tbl.antibiotic_time IS NOT NULL
           AND
               (
                 -- if charttime is available, use it
                   (
                       ab_tbl.antibiotic_time >= me.charttime - INTERVAL '24' HOUR
                   AND ab_tbl.antibiotic_time <= me.charttime + INTERVAL '72' HOUR

                   )
                OR
                   (
                  -- if charttime is not available, use chartdate
                        me.charttime IS NULL
                    AND ab_tbl.antibiotic_time >= me.chartdate - INTERVAL '24' HOUR
                    AND ab_tbl.antibiotic_time < me.chartdate + INTERVAL '96' HOUR -- Note this is 96 hours to include cases of when antibiotics are given 3 days after chart date of culture
                    )
                )
        )

        -- select only the unique times for suspicion of infection
       , unique_times AS 
       (
        SELECT icustay_id, t_suspicion, COUNT(*) AS repeats FROM ab_fnl
         GROUP BY icustay_id, t_suspicion
         ORDER BY icustay_id, t_suspicion
        )

        -- Around each suspicion of infection, check the changes of the current SOFA score from the beginning of the window

       , sofa_scores AS
       (
        SELECT --psofa.*, t_suspicion
               u.icustay_id, hr, starttime, endtime, t_suspicion, sofa_24hours
               , FIRST_VALUE(sofa_24hours) OVER 
                     (PARTITION BY psofa.icustay_id, t_suspicion 
                          ORDER BY psofa.icustay_id, t_suspicion, starttime) AS initial_sofa
               , sofa_24hours - FIRST_VALUE(sofa_24hours) OVER 
                     (PARTITION BY psofa.icustay_id, t_suspicion 
                          ORDER BY psofa.icustay_id, t_suspicion, starttime) AS sofa_difference
          FROM unique_times u
          LEFT JOIN pivoted_sofa_mod psofa
            ON u.icustay_id = psofa.icustay_id
    
    
        -- Specify the sensitivity needed
         WHERE (starttime <= t_suspicion + INTERVAL '12' HOUR 
           AND starttime >= t_suspicion - INTERVAL '24' HOUR) 

        -- If in the future we need to compute t_sofa even for patients without a t_suspicion then uncomment the following    
        --    OR t_suspicion IS NULL
         ORDER BY psofa.icustay_id, t_suspicion, starttime
        )
        

        -- find where the SOFA score has increased by 2 within the time sensitivity being investigated
        -- Note the sepsis-3 papers mentions that the baseline of a patient should be zero so an alternative is just to test if the total SOFA score exceeds 2. However that approach would have lower specificity and may be less clinically interesting

       , sofa_times AS 
       (
        SELECT icustay_id, t_suspicion, MIN(starttime) AS t_sofa
          FROM sofa_scores
         WHERE sofa_difference >= 2
         GROUP BY icustay_id, t_suspicion
        )

        -- Find the first time when the sepsis-3 requirements are satisfied
       , first_event AS
       (
        SELECT icustay_id, MIN(t_suspicion) AS t_suspicion, MIN(t_sofa) AS t_sofa 
          FROM sofa_times
         GROUP BY icustay_id
        )

        -- using this method we only list t_suspicion if it was paired with t_sofa, can adapt this script to find individual times
        -- use unique times if need to retain t_suspicion
       , cohort AS
       (
        SELECT subject_id, hadm_id, fil.icustay_id, intime, outtime
               , t_suspicion, t_sofa 
               , LEAST(t_suspicion, t_sofa) AS t_sepsis_min
               , GREATEST(t_suspicion, t_sofa) AS t_sepsis_max
          FROM filtered_ids fil
         INNER JOIN icustays ie ON fil.icustay_id = ie.icustay_id
          LEFT JOIN first_event fe ON fil.icustay_id = fe.icustay_id
         ORDER BY subject_id, intime
        )

SELECT * FROM cohort
