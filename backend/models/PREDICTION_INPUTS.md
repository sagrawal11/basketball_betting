# Prediction Engine - Additional Inputs for Better Context

## 🎯 Current Inputs (What We Have Now)
- ✅ Player name
- ✅ Opponent team
- ✅ Home/Away
- ✅ Game date
- ✅ Season

## 🚀 Additional Inputs (HIGH IMPACT - Easy to Add)

### 1. **Injury Report / Player Status**
```python
injury_status: str  # "Healthy", "Probable", "Questionable", "Out"
minutes_restriction: bool  # Coming back from injury?
```
**Impact**: 🔥🔥🔥 HUGE! Injured players perform worse  
**Source**: NBA injury reports API

### 2. **Expected Minutes**
```python
expected_minutes: float  # 35, 28, 15, etc.
starter: bool  # Starting vs coming off bench
```
**Impact**: 🔥🔥🔥 HUGE! More minutes = more stats  
**Source**: Recent game averages, depth chart

### 3. **Teammate Availability**
```python
teammates_out: list  # ["Anthony Davis", "D'Angelo Russell"]
star_teammates_playing: bool
```
**Impact**: 🔥🔥 HIGH! When AD is out, LeBron shoots more  
**Source**: Injury reports

### 4. **Rest Days (Enhanced)**
```python
days_rest: int  # Auto-calculated but could override
games_in_last_7_days: int
is_3rd_game_in_4_nights: bool
travel_distance: float  # Miles traveled
```
**Impact**: 🔥🔥 HIGH! Fatigue affects performance  
**Source**: Schedule data

### 5. **Game Importance**
```python
game_importance: str  # "Regular", "Rivalry", "Playoff Race", "Finals"
playoff_game: bool
season_game_number: int  # Game 5 vs Game 75
```
**Impact**: 🔥 MEDIUM-HIGH! Players try harder in important games  
**Source**: Standings, calendar

---

## 💡 Additional Inputs (MEDIUM IMPACT - Moderate Effort)

### 6. **Specific Defensive Matchup**
```python
primary_defender: str  # Who guards him? "Alex Caruso"
defender_quality: str  # "Elite", "Above Average", "Average", "Below Average"
```
**Impact**: 🔥🔥 HIGH! Elite defenders limit scoring  
**Source**: Matchup data, defensive stats

### 7. **Referee Assignment**
```python
referee_crew: list  # ["Scott Foster", "Tony Brothers"]
foul_call_tendency: str  # "Tight", "Average", "Lenient"
```
**Impact**: 🔥 MEDIUM! Some refs call more fouls = more FTs  
**Source**: NBA official assignments

### 8. **Recent Form / Streaks**
```python
hot_streak: bool  # 30+ points last 3 games
cold_streak: bool  # Under 20 points last 3 games
recent_trend: str  # "Improving", "Declining", "Stable"
```
**Impact**: 🔥 MEDIUM! Momentum matters  
**Source**: Already have this data! Just need to expose it

### 9. **Game Time / TV**
```python
game_time: str  # "7:30 PM ET", "12:00 PM ET"
nationally_televised: bool  # ESPN, TNT, ABC
```
**Impact**: 💡 LOW-MEDIUM! Stars perform better on national TV  
**Source**: Schedule data

### 10. **Altitude**
```python
altitude: str  # "Denver" (5280 ft) vs sea level
```
**Impact**: 💡 MEDIUM! Denver games have higher scoring  
**Source**: City data

---

## 🔬 Advanced Inputs (HIGH IMPACT - Harder to Get)

### 11. **Vegas Insights**
```python
vegas_total_line: float  # Over/under for game total
point_spread: float  # LAL -5.5
implied_team_total: float  # Lakers projected 110 points
```
**Impact**: 🔥🔥 HIGH! Vegas knows something we don't  
**Source**: SportsGameOdds API (you have this!)

### 12. **Lineup Data**
```python
starting_lineup: list  # All 5 starters
bench_strength: float  # Quality of bench unit
```
**Impact**: 🔥🔥 HIGH! Playing with good teammates = more assists  
**Source**: Rotowire, team announcements

### 13. **Advanced Matchup Stats**
```python
opponent_vs_position: dict  # How Bulls defend against PFs
opponent_pace_last_10: float  # Recent pace, not season average
opponent_injuries: list  # Their key players out
```
**Impact**: 🔥🔥 HIGH! Recent form > season averages  
**Source**: NBA stats API

### 14. **Betting Market Data**
```python
line_movement: str  # "Moving up", "Moving down", "Stable"
public_betting_percentage: float  # 75% on over
sharp_money: str  # Where sharp bettors are
```
**Impact**: 🔥🔥🔥 HUGE for betting! Market inefficiencies  
**Source**: SportsGameOdds, betting sites

### 15. **Real-Time Updates**
```python
injury_report_time: datetime  # When was report last updated
weather: dict  # For outdoor events (not NBA)
court_conditions: str  # New floor, rim tightness
```
**Impact**: 💡 LOW for NBA, HIGH for other sports

---

## 📊 What I Recommend Adding FIRST:

### **Tier 1 (Must Have - Easy Wins):**
1. ✅ **Expected minutes** - Huge impact, easy to estimate
2. ✅ **Injury status** - Critical for accuracy
3. ✅ **Teammates out** - Changes usage rate
4. ✅ **Vegas total line** - You already have access!

### **Tier 2 (Should Have - Worth the Effort):**
5. ✅ **Primary defender** - Affects scoring
6. ✅ **Game importance** - Playoff race games differ
7. ✅ **Recent opponent pace** - Better than season average
8. ✅ **Starting vs bench** - Role matters

### **Tier 3 (Nice to Have - Advanced):**
9. Referee crew
10. Lineup combinations
11. Betting market signals
12. Altitude adjustment

---

## 💡 Example Enhanced Prediction Call:

```python
preds = predictor.predict_player_stats(
    # BASIC (current)
    player_name='LeBron James',
    opponent_team='CHI',
    is_home=False,
    game_date='2025-10-15',
    season='2025-26',
    
    # TIER 1 (high impact)
    expected_minutes=35.0,
    injury_status='Healthy',
    teammates_out=['Anthony Davis'],  # AD injured
    vegas_game_total=218.5,
    
    # TIER 2 (medium impact)
    primary_defender='Alex Caruso',
    is_starter=True,
    game_importance='Rivalry',
    opponent_pace_last_10=102.5,  # Bulls playing fast recently
    
    # TIER 3 (nice to have)
    referee_crew=['Scott Foster'],
    nationally_televised=True,
    altitude='Normal'
)
```

Would you like me to implement the **Tier 1 inputs** (expected minutes, injury status, teammates out, vegas total)? These would significantly improve prediction accuracy!

