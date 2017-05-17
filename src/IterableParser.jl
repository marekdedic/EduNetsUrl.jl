import Base: start, next, done;
import StatsBase.sample;

export IterableParser, start, next, done, sample;

type IterableParser
	urls::Vector{AbstractString};
	labels::Vector{Int};
	batchSize::Int;
	processor::Function;

	featureCount::Int;
	featureGenerator::Function;
	T::DataType;
end

function IterableParser(urls::Vector{AbstractString}, labels::Vector{Int}, batchSize::Int; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, parallel::Bool = false, T::DataType = Float32)
	if parallel
		processor = processDatasetParallel;
	else
		processor = processDataset;
	end
	return IterableParser(urls, labels, batchSize, processor, featureCount, featureGenerator, T);
end

function start(::IterableParser)::Int
	return 1;
end

function next(iter::IterableParser, state::Int)::Tuple{Dataset, Int}
	if iter.batchSize == 0
		start = 1;
		stop = length(iter.labels);
	else
		start = (state - 1) * iter.batchSize + 1;
		stop = min(state * iter.batchSize, length(iter.labels));
	end
	dataset = iter.processor(iter.urls[start:stop], iter.labels[start:stop]; featureCount = iter.featureCount, featureGenerator = iter.featureGenerator, T = iter.T);
	return (dataset, state + 1);
end

function done(iter::IterableParser, state::Int)::Bool
	if iter.batchSize == 0
		return state > 1;
	end
	return state > cld(length(iter.labels), iter.batchSize);
end

function sample(iter::IterableParser)
	perm = sample(1:length(iter.urls), iter.batchSize);
	return iter.processor(iter.urls[perm], iter.labels[perm]; featureCount = iter.featureCount, featureGenerator = iter.featureGenerator, T = iter.T);
end
