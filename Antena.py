import math

def calculate_base_station_coverage_radius(frequency_mhz, power_dbm, height_bs_m, standard ='4G' , environment='urban', terrain='flat'):
    """
    Расчет радиуса покрытия базовой станции мобильной сети
    
    Параметры:
    - frequency_mhz: частота в МГц
    - power_dbm: мощность передатчика в дБм  
    - height_bs_m: высота базовой станции в метрах
    - standard: стандарт связи ('2G', '3G', '4G')
    - environment: тип местности ('urban', 'suburban', 'rural')
    - terrain: рельеф ('flat', 'hilly', 'mountainous')
    
    Возвращает радиус покрытия в км
    """
    
    # Параметры приемника для разных стандартов
    receiver_params = {
        '2G': {'sensitivity': -102, 'margin': 8, 'max_radius': 35},
        '3G': {'sensitivity': -121, 'margin': 6, 'max_radius': 15},
        '4G': {'sensitivity': -101.5, 'margin': 8, 'max_radius': 20}
    }
    
    # Поправки для типа местности
    environment_corrections = {
        'urban': {'loss': 0, 'K': 3, 'radius_factor': 0.7},
        'suburban': {'loss': -3, 'K': 0, 'radius_factor': 1.0},
        'rural': {'loss': -6, 'K': 0, 'radius_factor': 1.5}
    }
    
    # Поправки для рельефа
    terrain_corrections = {
        'flat': 0, 'hilly': 3, 'mountainous': 8
    }
    
    if standard not in receiver_params:
        raise ValueError("Стандарт должен быть '2G', '3G' или '4G'")
    
    params = receiver_params[standard]
    env_corr = environment_corrections[environment]
    terrain_corr = terrain_corrections[terrain]
    
    # Системные параметры
    tx_antenna_gain = 18    # дБи
    rx_antenna_gain = 0     # дБи
    tx_losses = 2           # дБ
    rx_losses = 2           # дБ
    height_ms = 1.5         # м
    
    # Расчет MAPL с поправками
    total_margin = params['margin'] + terrain_corr
    mapl = (power_dbm + tx_antenna_gain - tx_losses + 
            rx_antenna_gain - rx_losses - total_margin - 
            params['sensitivity'] + env_corr['loss'])
    
    # Выбор модели распространения
    if frequency_mhz <= 1500:
        # Модель Окамура-Хата
        a_hms = (1.1 * math.log10(frequency_mhz) - 0.7 * height_ms - 
                (1.56 * math.log10(frequency_mhz) - 0.8))
        A = (69.55 + 26.16 * math.log10(frequency_mhz) - 
             13.82 * math.log10(height_bs_m) - a_hms)
        B = 44.9 - 6.55 * math.log10(height_bs_m)
    else:
        # Модель COST231-Хата
        a_hms = (1.1 * math.log10(frequency_mhz) - 0.7 * height_ms - 
                (1.56 * math.log10(frequency_mhz) - 0.8))
        A = (46.3 + 33.9 * math.log10(frequency_mhz) - 
             13.82 * math.log10(height_bs_m) - a_hms + env_corr['K'])
        B = 44.9 - 6.55 * math.log10(height_bs_m)
    
    # Расчет радиуса
    radius_km = 10 ** ((mapl - A) / B)
    
    # Применение ограничений
    max_radius = params['max_radius'] * env_corr['radius_factor']
    radius_km = min(radius_km, max_radius)
    radius_km = max(radius_km, 0.5)  # минимум 500м
    
    return radius_km

# Примеры использования
if __name__ == "__main__":
    # Пример 1: Типичная городская 4G станция
    radius = calculate_base_station_coverage_radius(
        frequency_mhz=1800, power_dbm=40, height_bs_m=30, 
        standard='4G', environment='urban', terrain='flat'
    )
    print(f"4G городская БС: {radius:.2f} км")
