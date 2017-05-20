using Base.Test;
using EduNets;

include("DatasetConsistency.jl");
include("featureGeneratorsConsistency.jl");
include("ModelConsistency.jl");

k = 20;
d = 1000;
o = 2;
T = Float32;
relumean = Model((ReluLayer((d, k); T = T), MeanPoolingLayer(k; T = T)),
				 (ReluLayer((d, k); T = T), MeanPoolingLayer(k; T = T)),
				 (ReluLayer((d, k); T = T), MeanPoolingLayer(k; T = T)),
				 (LinearLayer((3 * k, o); T = T), ));
relurelumean = Model((ReluLayer((d, k); T = T), ReluLayer((k, k); T = T), MeanPoolingLayer(k; T = T)),
					 (ReluLayer((d, k); T = T), ReluLayer((k, k); T = T), MeanPoolingLayer(k; T = T)),
					 (ReluLayer((d, k); T = T), ReluLayer((k, k); T = T), MeanPoolingLayer(k; T = T)),
					 (LinearLayer((3 * k, o); T = T), ));
maxout3mean = Model((MaxOutLayer((d, k), 3; T = T), MeanPoolingLayer(k; T = T)),
					(MaxOutLayer((d, k), 3; T = T), MeanPoolingLayer(k; T = T)),
					(MaxOutLayer((d, k), 3; T = T), MeanPoolingLayer(k; T = T)),
					(LinearLayer((3 * k, o); T = T), ));
maxoutmaxout3mean = Model((MaxOutLayer((d, k), 3; T = T), MaxOutLayer((k, k), 3; T = T), MeanPoolingLayer(k; T = T)),
						  (MaxOutLayer((d, k), 3; T = T), MaxOutLayer((k, k), 3; T = T), MeanPoolingLayer(k; T = T)),
						  (MaxOutLayer((d, k), 3; T = T), MaxOutLayer((k, k), 3; T = T), MeanPoolingLayer(k; T = T)),
						  (LinearLayer((3 * k, o); T = T), ));
@testset "All" begin
	@testset "Feature generator consistency" begin
		@testset "unigramFeatureGenerator" begin
			for i in 1:400
				@test testGenerator(unigramFeatureGenerator);
			end
		end
		@testset "bigramFeatureGenerator" begin
			for i in 1:400
				@test testGenerator(bigramFeatureGenerator);
			end
		end
		@testset "trigramFeatureGenerator" begin
			for i in 1:400
				@test testGenerator(trigramFeatureGenerator);
			end
		end
	end
	@testset "Dataset consistency" begin
		@testset "Unsorted" begin
			for i in 1:500
				@test testDataset(false);
			end
		end
		@testset "Sorted" begin
			for i in 1:500
				@test testDataset(true);
			end
		end
	end
	@testset "Model consistency" begin
		@testset "relumean" begin
			for i in 1:100
				@test testModel(relumean);
			end
		end
		@testset "relurelumean" begin
			for i in 1:100
				@test testModel(relurelumean);
			end
		end
		@testset "maxout3mean" begin
			for i in 1:100
				@test testModel(maxout3mean);
			end
		end
		@testset "maxoutmaxout3mean" begin
			for i in 1:100
				@test testModel(maxoutmaxout3mean);
			end
		end
	end
end
