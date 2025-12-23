"""
main.py
Main execution script for MLB WAR & Contract Value Analysis
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

from war_data_fetcher import WARDataFetcher
from contract_data_fetcher import ContractDataFetcher
from war_processor import WARProcessor
from value_model import PlayerValueModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlb_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLBAnalysisPipeline:
    """Main pipeline for MLB WAR and contract analysis"""
    
    def __init__(self, data_dir: Path = Path("data"), 
                 output_dir: Path = Path("outputs")):
        """
        Initialize pipeline
        
        Args:
            data_dir: Directory for data storage
            output_dir: Directory for outputs
        """
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.output_dir = output_dir
        
        # Create directories
        for d in [self.raw_dir, self.processed_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized MLB Analysis Pipeline")
    
    def fetch_all_data(self, start_year: int = 1985, force_refresh: bool = False):
        """
        Fetch all required data
        
        Args:
            start_year: Starting year for data collection
            force_refresh: Force re-download even if data exists
        """
        logger.info(f"Fetching data from {start_year}...")
        
        # Check if data already exists
        war_file = self.raw_dir / "war_baseball_reference.parquet"
        if war_file.exists() and not force_refresh:
            logger.info("WAR data already exists. Use --force-refresh to re-download.")
        else:
            # Fetch WAR data
            war_fetcher = WARDataFetcher(self.raw_dir)
            
            logger.info("Fetching Baseball-Reference WAR data...")
            bref_war = war_fetcher.fetch_baseball_reference_war(start_year)
            war_fetcher.save_data(bref_war, "war_baseball_reference")
            
            logger.info("Fetching Lahman database...")
            lahman_data = war_fetcher.fetch_lahman_database()
            for name, df in lahman_data.items():
                war_fetcher.save_data(df, f"lahman_{name}")
        
        # Fetch contract data
        contract_file = self.raw_dir / "contracts_arbitration.parquet"
        if contract_file.exists() and not force_refresh:
            logger.info("Contract data already exists. Use --force-refresh to re-download.")
        else:
            contract_fetcher = ContractDataFetcher(self.raw_dir)
            
            # Process Lahman salaries
            try:
                lahman_salaries = pd.read_parquet(self.raw_dir / "lahman_salaries.parquet")
                salary_df = contract_fetcher.fetch_lahman_salaries(lahman_salaries, start_year)
                
                # Separate by type
                arb_df, fa_df = contract_fetcher.separate_contract_types(salary_df)
                
                contract_fetcher.save_data(arb_df, "contracts_arbitration")
                contract_fetcher.save_data(fa_df, "contracts_free_agency")
                
            except FileNotFoundError:
                logger.warning("Lahman salary data not found. Run WAR fetch first.")
    
    def process_war_data(self):
        """Process and aggregate WAR data"""
        logger.info("Processing WAR data...")
        
        # Load data
        war_file = self.raw_dir / "war_baseball_reference.parquet"
        if not war_file.exists():
            logger.error("WAR data not found. Run fetch step first.")
            return
        
        war_data = pd.read_parquet(war_file)
        
        # Load position data if available
        position_file = self.raw_dir / "lahman_appearances.parquet"
        if position_file.exists():
            lahman_appearances = pd.read_parquet(position_file)
            # Extract positions for each year
            # This is simplified; full implementation would merge properly
            position_data = lahman_appearances[['playerID', 'yearID']].copy()
            position_data.columns = ['player_id', 'year']
        else:
            position_data = None
        
        # Process
        processor = WARProcessor(war_data, position_data)
        
        # Save aggregations
        processor.save_aggregations(self.processed_dir)
        
        # Create aging curves
        curves = processor.create_war_curves()
        for name, df in curves.items():
            output_file = self.processed_dir / f"{name}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {name}")
        
        # Save enhanced WAR data with career year
        enhanced_war = processor.add_career_year()
        enhanced_war.to_parquet(self.processed_dir / "war_enhanced.parquet", index=False)
        
        logger.info("WAR processing complete")
    
    def calculate_player_values(self):
        """Calculate player value metrics"""
        logger.info("Calculating player values...")
        
        # Load data
        try:
            war_data = pd.read_parquet(self.processed_dir / "war_enhanced.parquet")
            arb_contracts = pd.read_parquet(self.raw_dir / "contracts_arbitration.parquet")
            fa_contracts = pd.read_parquet(self.raw_dir / "contracts_free_agency.parquet")
        except FileNotFoundError as e:
            logger.error(f"Required data not found: {e}")
            return
        
        # Create model
        model = PlayerValueModel(war_data, arb_contracts, fa_contracts)
        
        # Calculate and save metrics
        model.save_value_metrics(self.output_dir)
        
        # Generate additional reports
        current_year = war_data['year'].max()
        
        # Market rates
        rates = model.calculate_dollars_per_war(current_year)
        logger.info(f"\nCurrent Market Rates ({current_year}):")
        logger.info(f"  Arbitration: ${rates.get('arbitration', 0):,.0f} per WAR")
        logger.info(f"  Free Agency: ${rates.get('free_agency', 0):,.0f} per WAR")
        
        # Top trade values
        top_values = model.rank_players_by_trade_value(current_year, top_n=100)
        top_values.to_csv(self.output_dir / f"top_trade_values_{current_year}.csv", index=False)
        
        logger.info(f"\nTop 10 Trade Values ({current_year}):")
        print(top_values.head(10)[['rank', 'name', 'trade_value_index', 
                                     'recent_war_avg', 'age', 'contract_type']])
        
        logger.info("Value calculation complete")
    
    def generate_reports(self):
        """Generate summary reports"""
        logger.info("Generating reports...")
        
        report_path = self.output_dir / "analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("MLB WAR & CONTRACT VALUE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Load key datasets
            try:
                war_data = pd.read_parquet(self.processed_dir / "war_enhanced.parquet")
                
                f.write(f"DATA SUMMARY\n")
                f.write("-" * 60 + "\n")
                f.write(f"Total player-seasons: {len(war_data):,}\n")
                f.write(f"Unique players: {war_data['player_id'].nunique():,}\n")
                f.write(f"Year range: {war_data['year'].min()} - {war_data['year'].max()}\n")
                f.write(f"Total WAR: {war_data['war'].sum():,.1f}\n")
                f.write(f"Average WAR per season: {war_data['war'].mean():.2f}\n\n")
                
                # Top career WAR
                career_war = war_data.groupby('player_id').agg({
                    'war': 'sum',
                    'name': 'first'
                }).reset_index()
                career_war = career_war.nlargest(20, 'war')
                
                f.write("TOP 20 PLAYERS BY CAREER WAR\n")
                f.write("-" * 60 + "\n")
                for i, row in career_war.iterrows():
                    f.write(f"{i+1:2d}. {row['name']:30s} {row['war']:6.1f}\n")
                
            except Exception as e:
                f.write(f"Error generating report: {e}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def run_full_pipeline(self, start_year: int = 1985, force_refresh: bool = False):
        """
        Run complete analysis pipeline
        
        Args:
            start_year: Starting year
            force_refresh: Force data refresh
        """
        logger.info("Starting full pipeline...")
        
        try:
            # Step 1: Fetch data
            self.fetch_all_data(start_year, force_refresh)
            
            # Step 2: Process WAR data
            self.process_war_data()
            
            # Step 3: Calculate values
            self.calculate_player_values()
            
            # Step 4: Generate reports
            self.generate_reports()
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='MLB WAR & Contract Value Analysis System'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=1985,
        help='Starting year for data collection (default: 1985)'
    )
    
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh of all data'
    )
    
    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Only fetch data, do not process'
    )
    
    parser.add_argument(
        '--process-only',
        action='store_true',
        help='Only process existing data'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Data directory (default: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output directory (default: ./outputs)'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = MLBAnalysisPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Execute
    if args.fetch_only:
        pipeline.fetch_all_data(args.start_year, args.force_refresh)
    elif args.process_only:
        pipeline.process_war_data()
        pipeline.calculate_player_values()
        pipeline.generate_reports()
    else:
        pipeline.run_full_pipeline(args.start_year, args.force_refresh)


if __name__ == "__main__":
    main()
