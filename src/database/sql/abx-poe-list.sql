/*
This file is written by Alistaire Johnson, the original can be found at:
https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/abx-poe-list.sql
*/

  DROP TABLE IF EXISTS abx_poe_list CASCADE;
CREATE TABLE abx_poe_list AS
  WITH t1 AS
        (
          SELECT drug, drug_name_generic
                , route
                , CASE
                  WHEN LOWER(drug) LIKE '%' || LOWER('adoxa') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ala-tet') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('alodox') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('amikacin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('amikin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('amoxicillin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('amoxicillin%clavulanate') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('clavulanate') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ampicillin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('augmentin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('avelox') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('avidoxy') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('azactam') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('azithromycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('aztreonam') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('axetil') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('bactocill') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('bactrim') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('bethkis') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('biaxin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('bicillin l-a') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cayston') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefazolin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cedax') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefoxitin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ceftazidime') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefaclor') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefadroxil') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefdinir') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefditoren') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefepime') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefotetan') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefotaxime') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefpodoxime') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefprozil') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ceftibuten') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ceftin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefuroxime ') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cefuroxime') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cephalexin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('chloramphenicol') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cipro') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ciprofloxacin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('claforan') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('clarithromycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cleocin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('clindamycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('cubicin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('dicloxacillin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('doryx') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('doxycycline') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('duricef') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('dynacin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ery-tab') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('eryped') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('eryc') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('erythrocin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('erythromycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('factive') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('flagyl') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('fortaz') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('furadantin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('garamycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('gentamicin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('kanamycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('keflex') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ketek') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('levaquin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('levofloxacin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('lincocin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('macrobid') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('macrodantin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('maxipime') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('mefoxin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('metronidazole') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('minocin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('minocycline') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('monodox') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('monurol') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('morgidox') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('moxatag') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('moxifloxacin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('myrac') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('nafcillin sodium') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('nicazel doxy 30') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('nitrofurantoin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('noroxin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ocudox') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('ofloxacin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('omnicef') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('oracea') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('oraxyl') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('oxacillin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('pc pen vk') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('pce dispertab') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('panixine') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('pediazole') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('penicillin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('periostat') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('pfizerpen') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('piperacillin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('tazobactam') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('primsol') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('proquin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('raniclor') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('rifadin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('rifampin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('rocephin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('smz-tmp') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('septra') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('septra ds') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('septra') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('solodyn') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('spectracef') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('streptomycin sulfate') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('sulfadiazine') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('sulfamethoxazole') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('trimethoprim') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('sulfatrim') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('sulfisoxazole') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('suprax') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('synercid') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('tazicef') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('tetracycline') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('timentin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('tobi') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('tobramycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('trimethoprim') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('unasyn') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('vancocin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('vancomycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('vantin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('vibativ') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('vibra-tabs') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('vibramycin') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('zinacef') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('zithromax') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('zmax') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('zosyn') || '%' THEN 1
                  WHEN LOWER(drug) LIKE '%' || LOWER('zyvox') || '%' THEN 1
                ELSE 0
                END AS antibiotic
            FROM prescriptions
           WHERE drug_type IN ('MAIN','ADDITIVE')
              -- we exclude routes via the eye, ears, or topically
             AND route NOT IN ('OU','OS','OD','AU','AS','AD', 'TP')
             AND route NOT ILIKE '%ear%'
             AND route NOT ILIKE '%eye%'
              -- we exclude certain types of antibiotics: topical creams, gels, desens, etc
             AND drug NOT ILIKE '%cream%'
             AND drug NOT ILIKE '%desensitization%'
             AND drug NOT ILIKE '%ophth oint%'
             AND drug NOT ILIKE '%gel%'
              -- other routes not sure about...
              -- for sure keep: ('IV','PO','PO/NG','ORAL', 'IV DRIP', 'IV BOLUS')
              -- ? VT, PB, PR, PL, NS, NG, NEB, NAS, LOCK, J TUBE, IVT
              -- ? IT, IRR, IP, IO, INHALATION, IN, IM
              -- ? IJ, IH, G TUBE, DIALYS
              -- ?? enemas??
        )
SELECT drug --, drug_name_generic
       , COUNT(*) AS numobs
  FROM t1
 WHERE antibiotic = 1
 GROUP BY drug --, drug_name_generic
 ORDER BY numobs DESC;