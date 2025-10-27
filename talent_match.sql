CREATE TABLE final_match_results AS
SELECT
  `COL 1` AS employee_id,
  `COL 2` AS fullname,
  `COL 14` AS rating,
  ROUND((`COL 21` / 108.5) * 100, 2) AS iq_match,
  ROUND((`COL 16` / 60) * 100, 2) AS pauli_match,
  ROUND((`COL 22` / 28) * 100, 2) AS gtq_match,
  ROUND(
      (COALESCE(`COL 21`,0)/108.5)*100*0.4 +
      (COALESCE(`COL 16`,0)/60)*100*0.3 +
      (COALESCE(`COL 22`,0)/28)*100*0.3,
  2) AS final_match_rate
FROM merged_dataset
WHERE `COL 21` IS NOT NULL;

SELECT
  employee_id,
  fullname,
  rating,
  iq_match,
  pauli_match,
  gtq_match,
  final_match_rate
FROM final_match_results
ORDER BY final_match_rate DESC
LIMIT 10;
