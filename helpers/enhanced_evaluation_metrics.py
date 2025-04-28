import numpy as np


class EnhancedEvaluationMetrics:
    """
    Class implementing enhanced evaluation metrics for cross-reference system.
    """

    def __init__(self):
        self.rbp_p = 0.8

    def calculate_metrics(self, results_details, at_k_values=[1, 3, 5, 10]):
        """
        Calculate all evaluation metrics based on detailed results
        """
        metrics = {}

        # Calculate standard hit rate metrics
        for k in at_k_values:
            metrics[f'hit_rate_{k}'] = self._calculate_hit_rate_k(results_details, k)

        # Calculate MRR (Mean Reciprocal Rank)
        metrics['mrr'] = self._calculate_mrr(results_details)

        # Calculate RBP (Rank-Biased Precision)
        metrics['rbp'] = self._calculate_rbp(results_details, self.rbp_p)
        metrics['vrbp'] = self._calculate_vrbp(results_details)

        # Calculate NFRR (Normalized First Relevant Rank)
        metrics['nfrr'] = self._calculate_nfrr(results_details)

        # Calculate average position
        positions = [detail['found_at_position'] for detail in results_details if detail['found_at_position'] > 0]
        if positions:
            metrics['avg_position'] = np.mean(positions)
            metrics['median_position'] = np.median(positions)

        return metrics

    def _calculate_hit_rate_k(self, results_details, k):
        """Calculate hit rate at k"""
        correct_at_k = sum(1 for detail in results_details if 0 < detail['found_at_position'] <= k)
        return correct_at_k / len(results_details) * 100 if results_details else 0

    def _calculate_mrr(self, results_details):
        """
        Calculate Mean Reciprocal Rank.
        """
        reciprocal_ranks = []

        for detail in results_details:
            position = detail['found_at_position']
            if position > 0:
                reciprocal_ranks.append(1.0 / position)
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    def _calculate_vrbp(self, results_details, p_ranges=None):
        """
        Calculate Variable-p Rank-Biased Precision without normalization factor
        """
        if p_ranges is None:
            p_ranges = {
                (1, 3): 0.95,
                (4, 6): 0.9,
                (7, 10): 0.85,
                (11, 20): 0.8,
                (21, float('inf')): 0.7
            }

        sorted_ranges = sorted(p_ranges.items(), key=lambda x: x[0][0])

        vrbp_scores = []

        for detail in results_details:
            position = detail['found_at_position']

            if position <= 0:
                vrbp_scores.append(0.0)
                continue

            # Calculate the cumulative discount
            weight = 1.0
            current_pos = 1

            for (start, end), p_value in sorted_ranges:
                if current_pos > position - 1:
                    break

                range_end = min(end, position - 1)
                positions_in_range = max(0, range_end - current_pos + 1)

                # Apply discount for all positions in this range
                weight *= p_value ** positions_in_range
                current_pos = range_end + 1

            vrbp_scores.append(weight)

        return np.mean(vrbp_scores) if vrbp_scores else 0

    def _calculate_rbp(self, results_details, p=0.8):
        """
        Calculate  Rank-Biased Precision without normalization factor
        """
        rbp_scores = []

        for detail in results_details:
            position = detail['found_at_position']

            if position > 0:
                rbp = p ** (position - 1)
            else:
                rbp = 0.0

            rbp_scores.append(rbp)

        return np.mean(rbp_scores) if rbp_scores else 0

    def _calculate_nfrr(self, results_details, default_max_rank=100):
        """
        Calculate Normalized First Relevant Rank using pair-specific maximum ranks
        """
        nfrr_scores = []

        for detail in results_details:
            position = detail['found_at_position']
            max_rank = detail.get('max_rank', default_max_rank)

            # Ensure max_rank is at least 1 to avoid division by zero
            max_rank = max(1, max_rank)

            if 0 < position <= max_rank:
                nfrr = 1.0 - ((position - 1) / max_rank)
            else:
                nfrr = 0.0

            nfrr_scores.append(nfrr)

        return np.mean(nfrr_scores) if nfrr_scores else 0


    def calculate_by_category(self, results_details):
        """
        Calculate metrics broken down by category
        """
        # Group results by category
        category_results = {}

        for detail in results_details:
            category = detail.get('category', 'Unknown')

            if category not in category_results:
                category_results[category] = []

            category_results[category].append(detail)

        # Calculate metrics for each category
        metrics_by_category = {}

        for category, details in category_results.items():
            metrics_by_category[category] = self.calculate_metrics(details)
            metrics_by_category[category]['total_pairs'] = len(details)

        return metrics_by_category

