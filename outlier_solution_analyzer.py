"""
Outlier Solution Analyzer

This script helps analyze outlier solutions by finding their corresponding:
- Construct ID and rank
- Spec ID and spec name  
- Prompt version/template used to generate the spec
- Performance metrics and comparison

Usage:
    python outlier_solution_analyzer.py --solution-id <solution_id>
    python outlier_solution_analyzer.py --optimization-id <optimization_id> --find-outliers
"""

import asyncio
import sys
import argparse
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import numpy as np

# Add current directory to path for imports
sys.path.append('.')

from benchmark_evaluator_meta_artemis import MetaArtemisEvaluator, LLMType
from meta_artemis_modules.project_manager import get_project_info_async, get_existing_solutions_async
from meta_artemis_modules.recommendations import get_top_construct_recommendations, get_top_ranked_constructs
from shared_templates import DEFAULT_PROJECT_OPTIMISATION_IDS, get_project_configurations
from loguru import logger


class OutlierSolutionAnalyzer:
    """Analyzer for finding and analyzing outlier solutions"""
    
    def __init__(self):
        self.project_configurations = get_project_configurations()
    
    async def analyze_solution(self, solution_id: str) -> Dict[str, Any]:
        """
        Analyze a specific solution to find its details and source information
        
        Args:
            solution_id: The solution ID to analyze
            
        Returns:
            Dictionary containing solution analysis results
        """
        logger.info(f"🔍 Analyzing solution: {solution_id}")
        
        # Find which project this solution belongs to
        project_id = await self._find_project_for_solution(solution_id)
        if not project_id:
            return {"error": f"Could not find project for solution {solution_id}"}
        
        project_config = self.project_configurations.get(project_id, {})
        project_name = project_config.get("name", "Unknown")
        
        logger.info(f"📊 Found solution in project: {project_name} ({project_id})")
        
        # Get project info and solutions
        try:
            project_info, existing_optimizations, existing_solutions = await get_existing_solutions_async(project_id)
            
            # Find the target solution
            target_solution = None
            for solution in existing_solutions:
                if solution.get("solution_id") == solution_id:
                    target_solution = solution
                    break
            
            if not target_solution:
                return {"error": f"Solution {solution_id} not found in project {project_name}"}
            
            # Extract solution details
            solution_specs = target_solution.get("specs", [])
            results_summary = target_solution.get("results_summary", {})
            
            logger.info(f"📋 Solution has {len(solution_specs)} specs")
            
            # Analyze each spec in the solution
            spec_analyses = []
            for spec in solution_specs:
                spec_analysis = await self._analyze_spec(
                    project_id=project_id,
                    project_name=project_name,
                    spec=spec,
                    results_summary=results_summary
                )
                spec_analyses.append(spec_analysis)
            
            # Get overall solution performance metrics
            performance_metrics = self._extract_performance_metrics(results_summary)
            
            analysis_result = {
                "solution_id": solution_id,
                "project_id": project_id,
                "project_name": project_name,
                "solution_name": target_solution.get("solution_name", "Unknown"),
                "optimization_name": target_solution.get("optimization_name", "Unknown"),
                "status": target_solution.get("status", "Unknown"),
                "created_at": target_solution.get("created_at", "Unknown"),
                "has_results": target_solution.get("has_results", False),
                "num_specs": len(solution_specs),
                "performance_metrics": performance_metrics,
                "spec_analyses": spec_analyses,
                "is_outlier": self._is_performance_outlier(performance_metrics)
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing solution: {str(e)}")
            return {"error": f"Error analyzing solution: {str(e)}"}
    
    async def find_outliers_in_optimization(self, optimization_id: str, outlier_threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Find outlier solutions in a specific optimization
        
        Args:
            optimization_id: The optimization ID to search
            outlier_threshold: Number of standard deviations to consider as outlier
            
        Returns:
            List of outlier solution analyses
        """
        logger.info(f"🔍 Finding outliers in optimization: {optimization_id}")
        
        # Find which project this optimization belongs to
        project_id = None
        for pid, opt_id in DEFAULT_PROJECT_OPTIMISATION_IDS.items():
            if opt_id == optimization_id:
                project_id = pid
                break
        
        if not project_id:
            return [{"error": f"Could not find project for optimization {optimization_id}"}]
        
        project_config = self.project_configurations.get(project_id, {})
        project_name = project_config.get("name", "Unknown")
        
        logger.info(f"📊 Found optimization in project: {project_name} ({project_id})")
        
        try:
            # Get all solutions in the optimization
            project_info, existing_optimizations, existing_solutions = await get_existing_solutions_async(project_id)
            
            # Filter solutions by optimization ID
            optimization_solutions = [
                sol for sol in existing_solutions 
                if sol.get("optimization_id") == optimization_id
            ]
            
            logger.info(f"📋 Found {len(optimization_solutions)} solutions in optimization")
            
            # Extract performance metrics for all solutions
            solution_metrics = []
            for solution in optimization_solutions:
                results_summary = solution.get("results_summary", {})
                performance_metrics = self._extract_performance_metrics(results_summary)
                
                if performance_metrics.get("runtime_avg") is not None:
                    solution_metrics.append({
                        "solution_id": solution.get("solution_id"),
                        "runtime_avg": performance_metrics["runtime_avg"],
                        "solution": solution
                    })
            
            if len(solution_metrics) < 2:
                return [{"error": "Not enough solutions with runtime data to detect outliers"}]
            
            # Calculate outlier threshold
            runtimes = [s["runtime_avg"] for s in solution_metrics]
            mean_runtime = np.mean(runtimes)
            std_runtime = np.std(runtimes)
            
            logger.info(f"📊 Runtime statistics: mean={mean_runtime:.3f}s, std={std_runtime:.3f}s")
            
            # Find outliers (solutions beyond threshold standard deviations)
            outliers = []
            for sol_metric in solution_metrics:
                runtime = sol_metric["runtime_avg"]
                z_score = abs(runtime - mean_runtime) / std_runtime if std_runtime > 0 else 0
                
                if z_score >= outlier_threshold:
                    logger.info(f"🚨 Found outlier: {sol_metric['solution_id'][:8]}... runtime={runtime:.3f}s (z-score={z_score:.2f})")
                    
                    # Analyze the outlier solution
                    outlier_analysis = await self.analyze_solution(sol_metric["solution_id"])
                    outlier_analysis["outlier_stats"] = {
                        "runtime": runtime,
                        "mean_runtime": mean_runtime,
                        "std_runtime": std_runtime,
                        "z_score": z_score,
                        "deviation_factor": runtime / mean_runtime if mean_runtime > 0 else float('inf')
                    }
                    outliers.append(outlier_analysis)
            
            logger.info(f"✅ Found {len(outliers)} outlier solutions")
            return outliers
            
        except Exception as e:
            logger.error(f"❌ Error finding outliers: {str(e)}")
            return [{"error": f"Error finding outliers: {str(e)}"}]
    
    async def _find_project_for_solution(self, solution_id: str) -> Optional[str]:
        """Find which project contains the given solution"""
        for project_id in self.project_configurations.keys():
            try:
                _, _, existing_solutions = await get_existing_solutions_async(project_id)
                for solution in existing_solutions:
                    if solution.get("solution_id") == solution_id:
                        return project_id
            except Exception as e:
                logger.debug(f"Error checking project {project_id}: {str(e)}")
                continue
        return None
    
    async def _analyze_spec(self, project_id: str, project_name: str, spec: Dict[str, Any], results_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific spec to find its source and details"""
        construct_id = spec.get("construct_id")
        spec_id = spec.get("spec_id")
        
        logger.debug(f"🔍 Analyzing spec {spec_id[:8]}... for construct {construct_id[:8]}...")
        
        # Get construct rank
        construct_rank = await self._get_construct_rank(project_id, construct_id)
        
        # Get spec source information (recommendation/template)
        spec_source = await self._get_spec_source(project_id, spec_id, construct_id)
        
        # Extract performance for this construct
        construct_performance = self._extract_construct_performance(results_summary, construct_id)
        
        return {
            "construct_id": construct_id,
            "construct_rank": construct_rank,
            "spec_id": spec_id,
            "spec_name": spec_source.get("spec_name", "Unknown"),
            "template_name": spec_source.get("template_name", "Unknown"),
            "template_id": spec_source.get("template_id", "Unknown"),
            "prompt_version": spec_source.get("prompt_version", "Unknown"),
            "ai_run_id": spec_source.get("ai_run_id", "Unknown"),
            "created_at": spec_source.get("created_at", "Unknown"),
            "source": spec_source.get("source", "Unknown"),
            "performance": construct_performance
        }
    
    async def _get_construct_rank(self, project_id: str, construct_id: str) -> Optional[int]:
        """Get the rank of a construct in the project"""
        try:
            # Setup evaluator to get ranked constructs
            evaluator = MetaArtemisEvaluator(
                task_name="runtime_performance",
                meta_prompt_llm_type=LLMType("gpt-4-o"),
                code_optimization_llm_type=LLMType("gpt-4-o"),
                project_id=project_id
            )
            
            await evaluator.setup_clients()
            
            # Get top-ranked constructs
            top_ranked_constructs = get_top_ranked_constructs(project_id, evaluator, top_n=10)
            
            # Find the rank of our construct
            for i, ranked_construct_id in enumerate(top_ranked_constructs):
                if ranked_construct_id == construct_id:
                    return i + 1  # Rank is 1-based
            
            return None  # Not in top 10
            
        except Exception as e:
            logger.debug(f"Error getting construct rank: {str(e)}")
            return None
    
    async def _get_spec_source(self, project_id: str, spec_id: str, construct_id: str) -> Dict[str, Any]:
        """Get the source information for a spec (which recommendation/template created it)"""
        try:
            # Get project info and specs
            project_info, project_specs, _ = await get_project_info_async(project_id)
            
            # Get top construct recommendations
            recommendations = get_top_construct_recommendations(
                project_id=project_id,
                project_specs=project_specs,
                generated_recommendations=None,
                top_n=10
            )
            
            # Find the recommendation that created this spec
            for rec in recommendations:
                if rec.get("spec_id") == spec_id or rec.get("ai_run_id") == spec_id:
                    # Determine prompt version based on template
                    template_name = rec.get("template_name", "")
                    template_id = rec.get("template_id", "")
                    
                    if "Enhanced" in template_name or template_id == "enhanced":
                        prompt_version = "Enhanced Meta-Prompt Template"
                    elif "Standard" in template_name or template_id == "standard":
                        prompt_version = "Standard Meta-Prompt Template"
                    elif "Simplified" in template_name or template_id == "simplified":
                        prompt_version = "Simplified Meta-Prompt Template"
                    elif "Baseline" in template_name or template_id == "baseline":
                        prompt_version = "Baseline Prompt (No Meta-Prompting)"
                    else:
                        prompt_version = f"Unknown Template: {template_name} (ID: {template_id})"
                    
                    return {
                        "spec_name": rec.get("spec_name", "Unknown"),
                        "template_name": template_name,
                        "template_id": template_id,
                        "prompt_version": prompt_version,
                        "ai_run_id": rec.get("ai_run_id", "Unknown"),
                        "created_at": rec.get("created_at", "Unknown"),
                        "source": rec.get("source", "Unknown")
                    }
            
            return {
                "spec_name": f"spec-{spec_id[:8]}...",
                "template_name": "Unknown",
                "template_id": "Unknown",
                "prompt_version": "Unknown",
                "ai_run_id": "Unknown",
                "created_at": "Unknown",
                "source": "Unknown"
            }
            
        except Exception as e:
            logger.debug(f"Error getting spec source: {str(e)}")
            return {
                "spec_name": "Error",
                "template_name": "Error",
                "template_id": "Error",
                "prompt_version": "Error",
                "ai_run_id": "Error",
                "created_at": "Error",
                "source": "Error"
            }
    
    def _extract_performance_metrics(self, results_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from solution results summary"""
        metrics = {
            "runtime_avg": None,
            "runtime_values": [],
            "runtime_count": 0
        }
        
        try:
            runtime_metrics = results_summary.get("runtime_metrics", {})
            if runtime_metrics:
                for metric_name, metric_data in runtime_metrics.items():
                    if isinstance(metric_data, dict):
                        avg_runtime = metric_data.get("avg")
                        values = metric_data.get("values", [])
                        
                        if avg_runtime is not None:
                            metrics["runtime_avg"] = float(avg_runtime)
                        
                        if values:
                            metrics["runtime_values"].extend([float(v) for v in values if isinstance(v, (int, float))])
                            metrics["runtime_count"] += len(values)
            
            # Fallback: look for direct runtime fields
            if metrics["runtime_avg"] is None:
                for key, value in results_summary.items():
                    if "runtime" in str(key).lower() and isinstance(value, (int, float)):
                        metrics["runtime_avg"] = float(value)
                        break
        
        except Exception as e:
            logger.debug(f"Error extracting performance metrics: {str(e)}")
        
        return metrics
    
    def _extract_construct_performance(self, results_summary: Dict[str, Any], construct_id: str) -> Dict[str, Any]:
        """Extract performance metrics for a specific construct"""
        # For now, return overall performance since construct-specific extraction is complex
        return self._extract_performance_metrics(results_summary)
    
    def _is_performance_outlier(self, performance_metrics: Dict[str, Any], threshold: float = 10.0) -> bool:
        """Determine if performance metrics indicate an outlier (simple heuristic)"""
        runtime_avg = performance_metrics.get("runtime_avg")
        if runtime_avg is None:
            return False
        
        # Simple heuristic: runtime > threshold seconds is considered outlier
        return runtime_avg > threshold
    
    def print_analysis_result(self, result: Dict[str, Any]):
        """Print analysis result in a formatted way"""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("🔍 OUTLIER SOLUTION ANALYSIS")
        print("=" * 80)
        print(f"📊 SOLUTION DETAILS:")
        print(f"   Solution ID: {result['solution_id']}")
        print(f"   Project: {result['project_name']} ({result['project_id']})")
        print(f"   Solution Name: {result['solution_name']}")
        print(f"   Optimization: {result['optimization_name']}")
        print(f"   Status: {result['status']}")
        print(f"   Created: {result['created_at']}")
        print(f"   Has Results: {result['has_results']}")
        print(f"   Number of Specs: {result['num_specs']}")
        
        # Performance metrics
        perf = result['performance_metrics']
        if perf['runtime_avg'] is not None:
            print(f"   Runtime Average: {perf['runtime_avg']:.3f} seconds")
            if perf['runtime_count'] > 0:
                print(f"   Runtime Measurements: {perf['runtime_count']}")
        
        if result.get('is_outlier'):
            print(f"   🚨 PERFORMANCE OUTLIER DETECTED!")
        
        # Outlier statistics if available
        if "outlier_stats" in result:
            stats = result["outlier_stats"]
            print(f"   📊 OUTLIER STATISTICS:")
            print(f"      Runtime: {stats['runtime']:.3f}s")
            print(f"      Mean Runtime: {stats['mean_runtime']:.3f}s")
            print(f"      Standard Deviation: {stats['std_runtime']:.3f}s")
            print(f"      Z-Score: {stats['z_score']:.2f}")
            print(f"      Deviation Factor: {stats['deviation_factor']:.1f}x")
        
        print()
        print(f"📋 SPEC ANALYSES:")
        
        for i, spec_analysis in enumerate(result['spec_analyses']):
            print(f"   Spec {i+1}:")
            print(f"      Construct ID: {spec_analysis['construct_id']}")
            print(f"      Construct Rank: {spec_analysis['construct_rank']}")
            print(f"      Spec ID: {spec_analysis['spec_id']}")
            print(f"      Spec Name: {spec_analysis['spec_name']}")
            print(f"      Template: {spec_analysis['template_name']}")
            print(f"      Template ID: {spec_analysis['template_id']}")
            print(f"      🎯 PROMPT VERSION: {spec_analysis['prompt_version']}")
            print(f"      AI Run ID: {spec_analysis['ai_run_id']}")
            print(f"      Created: {spec_analysis['created_at']}")
            print(f"      Source: {spec_analysis['source']}")
            print()


async def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Analyze outlier solutions")
    parser.add_argument("--solution-id", help="Specific solution ID to analyze")
    parser.add_argument("--optimization-id", help="Optimization ID to search for outliers")
    parser.add_argument("--find-outliers", action="store_true", help="Find outliers in optimization")
    parser.add_argument("--threshold", type=float, default=3.0, help="Outlier threshold (standard deviations)")
    
    args = parser.parse_args()
    
    analyzer = OutlierSolutionAnalyzer()
    
    if args.solution_id:
        # Analyze specific solution
        result = await analyzer.analyze_solution(args.solution_id)
        analyzer.print_analysis_result(result)
    
    elif args.optimization_id and args.find_outliers:
        # Find outliers in optimization
        outliers = await analyzer.find_outliers_in_optimization(args.optimization_id, args.threshold)
        
        if outliers:
            print(f"🚨 Found {len(outliers)} outlier solutions:")
            print()
            for i, outlier in enumerate(outliers):
                print(f"--- OUTLIER {i+1} ---")
                analyzer.print_analysis_result(outlier)
                print()
        else:
            print("✅ No outliers found in the optimization")
    
    else:
        print("❌ Please provide either --solution-id or --optimization-id with --find-outliers")
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 