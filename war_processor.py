import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WARProcessor:
    """Process and aggregate WAR data"""
    
    def __init__(self, war_data: pd.DataFrame, position_data: pd.DataFrame = None):
        """
        Initialize processor
        
        Args:
            war_data: DataFrame with player WAR data
            position_data: Optional DataFrame with player positions
        """
        self.war_data = war_data.copy()
        self.position_data = position_data
        
        # Ensure required columns
        required_cols = ['year', 'player_id', 'war']
        if not all(col in self.war_data.columns for col in required_cols):
            raise ValueError(f"WAR data must contain: {required_cols}")
        
        logger.info(f"Initialized with {len(self.war_data)} WAR records")
    
    def calculate_rookie_years(self) -> pd.DataFrame:
        """
        Calculate rookie year for each player
        
        Returns:
            DataFrame with player_id and rookie_year
        """
        logger.info("Calculating rookie years...")
        
        rookie_years = self.war_data.groupby('player_id')['year'].min().reset_index()
        rookie_years.columns = ['player_id', 'rookie_year']
        
        logger.info(f"Calculated rookie years for {len(rookie_years)} players")
        return rookie_years
    
    def add_career_year(self) -> pd.DataFrame:
        """
        Add career year (years since rookie year) to WAR data
        
        Returns:
            WAR data with career_year column
        """
        logger.info("Adding career year column...")
        
        rookie_years = self.calculate_rookie_years()
        
        # Merge rookie years
        df = self.war_data.merge(rookie_years, on='player_id', how='left')
        
        # Calculate career year
        df['career_year'] = df['year'] - df['rookie_year']
        
        # Ensure non-negative
        df['career_year'] = df['career_year'].clip(lower=0)
        
        logger.info("Career year added successfully")
        return df
    
    def aggregate_by_position(self) -> pd.DataFrame:
        """
        Aggregate WAR by position and season
        
        Returns:
            DataFrame with position, season, and aggregated WAR statistics
        """
        logger.info("Aggregating WAR by position...")
        
        if self.position_data is not None:
            # Merge position data
            df = self.war_data.merge(
                self.position_data[['player_id', 'year', 'position']], 
                on=['player_id', 'year'], 
                how='left'
            )
        else:
            df = self.war_data.copy()
        
        # Check if position column exists
        if 'position' not in df.columns:
            logger.warning("No position data available")
            return pd.DataFrame()
        
        # Aggregate
        position_agg = df.groupby(['position', 'year']).agg({
            'war': ['sum', 'mean', 'median', 'std', 'count'],
            'player_id': 'count'
        }).reset_index()
        
        # Flatten columns
        position_agg.columns = [
            'position', 'season', 'sum_war', 'mean_war', 
            'median_war', 'std_war', 'count_war', 'num_players'
        ]
        
        logger.info(f"Aggregated {len(position_agg)} position-season combinations")
        return position_agg
    
    def aggregate_by_age(self) -> pd.DataFrame:
        """
        Aggregate WAR by age and season
        
        Returns:
            DataFrame with age, season, and aggregated WAR statistics
        """
        logger.info("Aggregating WAR by age...")
        
        if 'age' not in self.war_data.columns:
            logger.warning("No age data available")
            return pd.DataFrame()
        
        # Remove rows with missing age
        df = self.war_data.dropna(subset=['age']).copy()
        
        # Aggregate
        age_agg = df.groupby(['age', 'year']).agg({
            'war': ['sum', 'mean', 'median', 'std', 'count'],
            'player_id': 'count'
        }).reset_index()
        
        # Flatten columns
        age_agg.columns = [
            'age', 'season', 'sum_war', 'mean_war', 
            'median_war', 'std_war', 'count_war', 'num_players'
        ]
        
        logger.info(f"Aggregated {len(age_agg)} age-season combinations")
        return age_agg
    
    def aggregate_by_career_year(self) -> pd.DataFrame:
        """
        Aggregate WAR by career year (years since rookie year)
        
        Returns:
            DataFrame with career_year, season, and aggregated WAR statistics
        """
        logger.info("Aggregating WAR by career year...")
        
        # Add career year
        df = self.add_career_year()
        
        # Aggregate
        career_agg = df.groupby(['career_year', 'year']).agg({
            'war': ['sum', 'mean', 'median', 'std', 'count'],
            'player_id': 'count'
        }).reset_index()
        
        # Flatten columns
        career_agg.columns = [
            'career_year', 'season', 'sum_war', 'mean_war', 
            'median_war', 'std_war', 'count_war', 'num_players'
        ]
        
        logger.info(f"Aggregated {len(career_agg)} career-year combinations")
        return career_agg
    
    def create_war_curves(self) -> Dict[str, pd.DataFrame]:
        """
        Create WAR aging curves by different dimensions
        
        Returns:
            Dictionary with aging curves by position, overall
        """
        logger.info("Creating WAR aging curves...")
        
        curves = {}
        
        # Overall aging curve
        age_curve = self.war_data.groupby('age')['war'].agg(['mean', 'median', 'std', 'count']).reset_index()
        curves['age_curve'] = age_curve
        
        # Career year curve
        df = self.add_career_year()
        career_curve = df.groupby('career_year')['war'].agg(['mean', 'median', 'std', 'count']).reset_index()
        curves['career_curve'] = career_curve
        
        # By position (if available)
        if self.position_data is not None and 'position' in self.war_data.columns:
            for pos in self.war_data['position'].dropna().unique():
                pos_data = self.war_data[self.war_data['position'] == pos]
                pos_curve = pos_data.groupby('age')['war'].agg(['mean', 'median', 'count']).reset_index()
                curves[f'age_curve_{pos}'] = pos_curve
        
        logger.info(f"Created {len(curves)} aging curves")
        return curves
    
    def calculate_replacement_level(self, percentile: float = 20) -> float:
        """
        Calculate replacement level WAR
        
        Args:
            percentile: Percentile to use for replacement level (default 20th)
            
        Returns:
            Replacement level WAR value
        """
        replacement = np.percentile(self.war_data['war'].dropna(), percentile)
        logger.info(f"Replacement level (P{percentile}): {replacement:.2f} WAR")
        return replacement
    
    def calculate_war_above_replacement(self, replacement_level: float = None) -> pd.DataFrame:
        """
        Calculate WAR above replacement
        
        Args:
            replacement_level: Custom replacement level (if None, calculates automatically)
            
        Returns:
            WAR data with WARP (WAR above replacement) column
        """
        if replacement_level is None:
            replacement_level = self.calculate_replacement_level()
        
        df = self.war_data.copy()
        df['warp'] = df['war'] - replacement_level
        df['warp'] = df['warp'].clip(lower=0)  # Can't be negative
        
        logger.info("Calculated WAR above replacement")
        return df
    
    def get_player_war_history(self, player_id: str) -> pd.DataFrame:
        """
        Get complete WAR history for a player
        
        Args:
            player_id: Player identifier
            
        Returns:
            Player's WAR history
        """
        player_data = self.war_data[self.war_data['player_id'] == player_id].copy()
        player_data = player_data.sort_values('year')
        
        # Add career totals
        if len(player_data) > 0:
            player_data['cumulative_war'] = player_data['war'].cumsum()
        
        return player_data
    
    def get_top_players_by_war(self, year: int = None, top_n: int = 50) -> pd.DataFrame:
        """
        Get top players by WAR
        
        Args:
            year: Season year (if None, uses career totals)
            top_n: Number of top players to return
            
        Returns:
            Top players DataFrame
        """
        if year is not None:
            df = self.war_data[self.war_data['year'] == year].copy()
            df = df.nlargest(top_n, 'war')
        else:
            # Career totals
            career_war = self.war_data.groupby('player_id').agg({
                'war': 'sum',
                'name': 'first',
                'year': ['min', 'max']
            }).reset_index()
            career_war.columns = ['player_id', 'career_war', 'name', 'first_year', 'last_year']
            career_war = career_war.nlargest(top_n, 'career_war')
            df = career_war
        
        return df
    
    def save_aggregations(self, output_dir: Path):
        """
        Save all aggregations to files
        
        Args:
            output_dir: Directory to save outputs
        """
        logger.info(f"Saving aggregations to {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each aggregation
        aggs = {
            'war_by_position': self.aggregate_by_position(),
            'war_by_age': self.aggregate_by_age(),
            'war_by_career_year': self.aggregate_by_career_year()
        }
        
        for name, df in aggs.items():
            if not df.empty:
                df.to_csv(output_dir / f"{name}.csv", index=False)
                df.to_parquet(output_dir / f"{name}.parquet", index=False)
                logger.info(f"Saved {name}: {len(df)} records")
        
        # Save curves
        curves = self.create_war_curves()
        for name, df in curves.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)


def main():
    """Example usage"""
    from pathlib import Path
    
    # Load data
    data_dir = Path("data/raw")
    war_file = data_dir / "war_baseball_reference.parquet"
    
    if not war_file.exists():
        logger.error("WAR data not found. Run war_data_fetcher.py first.")
        return
    
    war_data = pd.read_parquet(war_file)
    
    # Process
    processor = WARProcessor(war_data)
    
    # Save aggregations
    output_dir = Path("data/processed")
    processor.save_aggregations(output_dir)
    
    # Print top players
    top_players = processor.get_top_players_by_war(top_n=20)
    print("\nTop 20 Players by Career WAR:")
    print(top_players[['name', 'career_war', 'first_year', 'last_year']])


if __name__ == "__main__":
    main()
