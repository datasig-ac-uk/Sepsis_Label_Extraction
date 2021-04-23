/* 
Based on the pivoted-vitals table (from MIT-LCP), in this script we create a pivoted view of the vital sign and chart data that we need
We define some bounds for each column to filter out clear anomalies
*/

  DROP TABLE IF EXISTS extracted_chart CASCADE;
CREATE TABLE extracted_chart AS

  WITH vitals AS (
                  SELECT c.subject_id, c.hadm_id, c.icustay_id, c.charttime
                         , MAX(CASE WHEN c.itemid IN (211,220045) 
                                     AND valuenum >= 0 
                                     AND valuenum <= 300 
                               THEN valuenum 
                               ELSE NULL END) AS heart_rate
                         , MAX(COALESCE(CASE WHEN c.itemid IN (212, 220048) 
                                        THEN value 
                                        ELSE NULL END)) AS heart_rhythm
                             -- highest recorded blood pressure was 370/360 therefore anything over 400 are anomalies
                         , MAX(CASE WHEN c.itemid IN (442, 455, 220179, 224167) 
                                     AND valuenum >= 0 
                                     AND valuenum < 400 
                               THEN valuenum 
                               ELSE NULL END) AS nbp_sys
                         , MAX(CASE WHEN c.itemid IN (8440, 8441, 220180) 
                                     AND valuenum >= 0 
                                     AND valuenum < 200 
                               THEN valuenum 
                               ELSE NULL END) AS nbp_dias
                         , MAX(CASE WHEN c.itemid IN (443, 456, 220181) 
                                     AND valuenum >= 0 
                                     AND valuenum < 266 
                               THEN valuenum 
                               ELSE NULL END) AS nbp_mean
                         , MAX(CASE WHEN c.itemid IN (51, 6701, 220050, 225309) 
                                     AND valuenum >= 0 
                                     AND valuenum < 400 
                               THEN valuenum 
                               ELSE NULL END) AS abp_sys
                         , MAX(CASE WHEN c.itemid IN (8364, 8368, 8555, 220051, 225310) 
                                     AND valuenum >= 0 
                                     AND valuenum < 200 
                               THEN valuenum 
                               ELSE NULL END) AS abp_dias
                         , MAX(CASE WHEN c.itemid IN (52, 6702, 220052, 225312) 
                                     AND valuenum >= 0 
                                     AND valuenum < 266 
                               THEN valuenum 
                               ELSE NULL END) AS abp_mean
                         , MAX(COALESCE(CASE WHEN c.itemid IN (674, 224642) 
                                        THEN value 
                                        ELSE NULL END)) AS temp_site
                         , MAX(CASE WHEN c.itemid IN (678, 3652, 3654, 227054, 223761) 
                                     AND valuenum > 59 
                                     AND valuenum < 113 
                               THEN (valuenum-32.0)*5.0/9.0 
                               WHEN c.itemid IN (676, 3655, 223762) 
                                     AND valuenum > 15 
                                     AND valuenum < 45  
                               THEN valuenum 
                               ELSE NULL END
                             -- Note 677 is calculated from 678, 679 calculated from 676 so not included
                              ) AS temp_celcius
                         , MAX(CASE WHEN c.itemid IN (646, 6719, 220277)
                                     AND valuenum > 0 
                                     AND valuenum <= 100 
                               THEN valuenum 
                               ELSE NULL END) AS o2sat
                         , MAX(CASE WHEN c.itemid IN (8113, 3603, 220210)  
                                     AND valuenum >= 0 
                                     AND valuenum < 150 
                               THEN valuenum 
                               ELSE NULL END) AS resp_rate
                         , MAX(CASE WHEN c.itemid IN (444, 224697) 
                                     AND valuenum >= 0 
                                     AND valuenum < 60 
                               THEN valuenum 
                               ELSE NULL END) AS mean_airway_pressure
                         , MAX(CASE WHEN c.itemid IN (535, 224695) 
                                     AND valuenum >= 0 
                                     AND valuenum < 90 
                               THEN valuenum 
                               ELSE NULL END) AS peak_insp_rate
                         , MAX(CASE WHEN c.itemid IN (543, 224696) 
                                     AND valuenum >= 0 
                                     AND valuenum < 80 
                               THEN valuenum 
                               ELSE NULL END) AS plateau_pressure
                         , MAX(CASE WHEN c.itemid IN (470, 471, 223834) 
                                     AND valuenum >= 0 
                                     AND valuenum <= 100 
                               THEN valuenum 
                               ELSE NULL END) AS o2flow
                         , MAX(COALESCE(CASE WHEN c.itemid IN (467, 468, 469, 3605, 226732) 
                                        THEN value 
                                        ELSE NULL END)) AS o2_device
                         , MAX(CASE WHEN c.itemid IN (614, 653, 1884, 224689, 224422) 
                                     AND valuenum >= 0 
                                     AND valuenum < 90 
                               THEN valuenum 
                               ELSE NULL END) AS resp_rate_spont
                         , MAX(CASE WHEN c.itemid IN (619, 224688) 
                                     AND valuenum >= 0 
                                     AND valuenum < 50 
                               THEN valuenum 
                               ELSE NULL END) AS resp_rate_set
                         , MAX(CASE WHEN c.itemid IN (615, 224690) 
                                     AND valuenum >= 0 
                                     AND valuenum < 150 
                               THEN valuenum 
                               ELSE NULL END) AS resp_rate_total
                         , MAX(CASE WHEN c.itemid IN (445, 448, 449, 450, 224687) 
                                     AND valuenum >= 0 
                                     AND valuenum < 40 
                               THEN valuenum 
                               ELSE NULL END) AS minute_vol           
                         , MAX(COALESCE(CASE WHEN c.itemid IN (723, 223900, 226758) 
                                             THEN value 
                                             ELSE NULL END)) AS verbal_response
                             -- Note the ids 223900, 226758 corresponds to the measurement that goes into the GCS
                         , MAX(COALESCE(CASE WHEN c.itemid IN (720, 223849) 
                                        THEN value 
                                        ELSE NULL END)) AS vent_mode
                         , MAX(CASE WHEN c.itemid IN (683, 224684) 
                                     AND valuenum >= 0 
                                     AND valuenum <= 2000 
                               THEN valuenum 
                               ELSE NULL END) AS tidal_vol_set
                         , MAX(CASE WHEN c.itemid IN (682, 224685) 
                                     AND valuenum >= 0 
                                     AND valuenum <= 2000 
                               THEN valuenum 
                               ELSE NULL END) AS tidal_vol_obs
                         , MAX(CASE WHEN c.itemid IN (654, 684, 3050, 3083, 224686) 
                                     AND valuenum >= 0 
                                     AND valuenum <= 5000 
                               THEN valuenum 
                               ELSE NULL END) AS tidal_vol_spon
                         , MAX(CASE WHEN c.itemid IN (506, 220339) 
                                     AND valuenum >= 0 
                                     AND valuenum < 50 
                               THEN valuenum 
                               ELSE NULL END) AS peep_set
                         , MAX(CASE WHEN c.itemid IN (189, 190, 2981, 7570)
                                     AND valuenum >= 0.20 
                                     AND valuenum <= 1.0 
                               THEN valuenum*100
                               WHEN c.itemid IN (191, 3420, 3422, 223835) 
                                     AND valuenum >= 20 
                                     AND valuenum <= 100 
                               THEN valuenum ELSE NULL END) AS fio2_chart
                         , MAX(CASE WHEN c.itemid IN (807, 811, 1529, 3745, 3744, 225664, 220621, 226537)
                                     AND valuenum > 0 
                                     AND valuenum < 1000 
                               THEN valuenum 
                               ELSE NULL END) AS glucose_chart           
                                           
                    FROM chartevents AS c
                   INNER JOIN d_items d ON c.itemid = d.itemid
                   WHERE c.itemid IN (
                                      211, 220045, 3450, 5815, 220047, 8518, 8549, 220046,
                                      212, 220048,
                                      442, 455, 220179, 224167,
                                      8440, 8441, 220180,
                                      443, 456, 220181,
                                      51, 6701, 220050, 225309,
                                      8364, 8368, 8555, 220051, 225310,
                                      52, 6702, 220052, 225312,
                                      674, 224642,
                                      678, 679, 3652, 3654, 6643, 227054, 223761,
                                      676, 677, 3655, 223762,
                                      646, 6719, 220277, 
                                      5820, 8554, 223769, 223770, 226253,
                                      618, 8113, 3603, 220210,
                                      819, 8553, 224161, 224162,
                                      444,224697,
                                      535, 224695,
                                      470, 471, 223834,
                                      467, 3605, 226732,
                                      614,653,1884,224689,224422,
                                      619, 224688,
                                      615, 224690,
                                      445, 448, 449, 450, 224687,
                                      723, 223900, 226758,
                                      720, 223849,
                                      682, 224684,
                                      683, 224685,
                                      654, 684, 3050, 3083, 224686,
                                      506, 220339,
                                      189, 190, 2981, 7570,
                                      191, 3420, 3422, 223835,
                                      807, 811, 1529, 3745, 3744, 225664, 220621, 226537
                                     )
                     -- exclude rows marked as error
                     AND (error IS NULL OR error = 0)
                   GROUP BY c.subject_id, c.hadm_id, c.icustay_id, c.charttime
                 )

SELECT * FROM vitals

