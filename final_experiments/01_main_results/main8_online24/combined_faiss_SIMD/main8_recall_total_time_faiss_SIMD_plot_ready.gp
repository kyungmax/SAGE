reset
set encoding utf8

set terminal pdfcairo enhanced color dashed size 14, 5.5 font "Helvetica,16"
set output "main8_recall_total_time_faiss_SIMD.pdf"

set datafile separator ","
set datafile missing ""
data_dir = "plot_ready_main8_recall_total_time_faiss_SIMD"

set border lw 1.5
set tics out nomirror font "Helvetica,14"
set grid xtics ytics mxtics mytics lc rgb "#D0D0D0" dt 3, lc rgb "#E5E5E5" dt 3

set style line 3 lt 1 dt 1 lw 3.5 lc rgb "#000000"  # FAISS-SIMD Vanilla
set style line 4 lt 1 dt 1 lw 3.5 lc rgb "#ff7f0e"  # FAISS-SIMD Ours
set style line 1 lt 1 dt 2 lw 3.5 lc rgb "#000000"  # hnswlib Vanilla
set style line 2 lt 1 dt 2 lw 3.5 lc rgb "#ff7f0e"  # hnswlib Ours
set style line 5 pt '★' ps 2.2 lw 1.0 lc rgb "black"

set multiplot layout 2,4 margins 0.05, 0.99, 0.2, 0.94 spacing 0.06, 0.14

set xlabel "Total search time (s, log scale)" offset 0, 0.2 font "Helvetica,15"
set ylabel "Recall\\@10" offset 1.5, 0 font "Helvetica,15" noenhanced
set logscale x
set format x "%g"
set format y "%.3f"
set mxtics 10
set mytics 2
set yrange [*:1.00]

# Plot-ready CSV columns:
# 1 ef
# 2 faiss_vanilla_time_s, 3 faiss_vanilla_recall
# 4 faiss_ours_time_s, 5 faiss_ours_recall
# 6 hnswlib_vanilla_time_s, 7 hnswlib_vanilla_recall
# 8 hnswlib_ours_time_s, 9 hnswlib_ours_recall
# 10 faiss_recommended_time_s, 11 faiss_recommended_recall

unset key

set title "NYTimes" font "Helvetica-Bold,16" offset 0, -0.5
plot data_dir . "/nytimes.csv" skip 1 using 2:3 with lines ls 3 notitle, \
     '' skip 1 using 4:5 with lines ls 4 notitle, \
     '' skip 1 using 6:7 with lines ls 1 notitle, \
     '' skip 1 using 8:9 with lines ls 2 notitle, \
     '' skip 1 using 10:11 with points ls 5 notitle

set title "GloVe100" font "Helvetica-Bold,16" offset 0, -0.5
plot data_dir . "/glove100.csv" skip 1 using 2:3 with lines ls 3 notitle, \
     '' skip 1 using 4:5 with lines ls 4 notitle, \
     '' skip 1 using 6:7 with lines ls 1 notitle, \
     '' skip 1 using 8:9 with lines ls 2 notitle, \
     '' skip 1 using 10:11 with points ls 5 notitle

set title "AGNews" font "Helvetica-Bold,16" offset 0, -0.5
plot data_dir . "/agnews.csv" skip 1 using 2:3 with lines ls 3 notitle, \
     '' skip 1 using 4:5 with lines ls 4 notitle, \
     '' skip 1 using 6:7 with lines ls 1 notitle, \
     '' skip 1 using 8:9 with lines ls 2 notitle, \
     '' skip 1 using 10:11 with points ls 5 notitle

set title "Landmark" font "Helvetica-Bold,16" offset 0, -0.5
plot data_dir . "/landmark.csv" skip 1 using 2:3 with lines ls 3 notitle, \
     '' skip 1 using 4:5 with lines ls 4 notitle, \
     '' skip 1 using 6:7 with lines ls 1 notitle, \
     '' skip 1 using 8:9 with lines ls 2 notitle, \
     '' skip 1 using 10:11 with points ls 5 notitle

set title "CohereWiki" font "Helvetica-Bold,16" offset 0, -0.5
plot data_dir . "/coherewiki.csv" skip 1 using 2:3 with lines ls 3 notitle, \
     '' skip 1 using 4:5 with lines ls 4 notitle, \
     '' skip 1 using 6:7 with lines ls 1 notitle, \
     '' skip 1 using 8:9 with lines ls 2 notitle, \
     '' skip 1 using 10:11 with points ls 5 notitle

set title "YouTube15M" font "Helvetica-Bold,16" offset 0, -0.5
plot data_dir . "/youtube15m.csv" skip 1 using 2:3 with lines ls 3 notitle, \
     '' skip 1 using 4:5 with lines ls 4 notitle, \
     '' skip 1 using 6:7 with lines ls 1 notitle, \
     '' skip 1 using 8:9 with lines ls 2 notitle, \
     '' skip 1 using 10:11 with points ls 5 notitle

set title "MSMARCOV1" font "Helvetica-Bold,16" offset 0, -0.5
plot data_dir . "/msmarcov1.csv" skip 1 using 2:3 with lines ls 3 notitle, \
     '' skip 1 using 4:5 with lines ls 4 notitle, \
     '' skip 1 using 6:7 with lines ls 1 notitle, \
     '' skip 1 using 8:9 with lines ls 2 notitle, \
     '' skip 1 using 10:11 with points ls 5 notitle

set title "SpaceV" font "Helvetica-Bold,16" offset 0, -0.5
set key at screen 0.5, 0.03 center center maxrows 2 samplen 2.5 spacing 1.2 font "Helvetica,14" box opaque width 2
plot data_dir . "/spacev.csv" skip 1 using 2:3 with lines ls 3 title "FAISS-SIMD Vanilla", \
     '' skip 1 using 4:5 with lines ls 4 title "FAISS-SIMD Ours", \
     '' skip 1 using 6:7 with lines ls 1 title "hnswlib Vanilla", \
     '' skip 1 using 8:9 with lines ls 2 title "hnswlib Ours", \
     '' skip 1 using 10:11 with points ls 5 title "FAISS-SIMD Recommended ef"

unset multiplot
set output
