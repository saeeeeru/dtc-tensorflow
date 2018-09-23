dataset='synthetic'

echo "-------------------------------------------"
echo "processed" $dataset "data"
echo "-------------------------------------------"

python train.py -d $dataset

echo "-------------------------------------------"
echo "finish Clustering"
echo "-------------------------------------------"