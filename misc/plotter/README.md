# plotter

* experienced_states_plotter.py
* running_average_plotter.py

### experienced_states_plotter
コツ度を経験した状態に関してのみ計算して描画する。（状態空間を離散化して、その１メモリに入る状態を経験していたら、その離散化空間は経験ずみ）

### running_average_plotter.py
* 時間に関して移動平均をとって、Q関数近似のノイズ削減を図る。
* 失敗軌道を除外するオプションもつける。