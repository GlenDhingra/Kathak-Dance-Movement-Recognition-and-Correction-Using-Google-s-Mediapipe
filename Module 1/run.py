from move_comparison import compare_positions
benchmark_video = './dance_videos/kathak1bench.mp4'
user_video = './dance_videos/kathak1right.mp4' # replace with 0 for webcam

error,acc,n = compare_positions(benchmark_video, user_video)
print("Overall Error:",error/n)
print("Overall Accuracy:",acc/n)