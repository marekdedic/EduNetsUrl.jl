import Base: Operators.getindex, vcat;
import EduNets.sample;
import EduNets;
import DataFrames;

export Dataset, getindex, vcat, sample;

type Dataset{T<:AbstractFloat}<:EduNets.AbstractDataset
	domains::EduNets.SortedSingleBagDataset{T}
	paths::EduNets.SortedSingleBagDataset{T}
	queries::EduNets.SortedSingleBagDataset{T}

	y::Vector{Int}
	info::DataFrames.DataFrame;
end

function Dataset{T<:AbstractFloat}(features::Matrix{T}, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; info::Vector{AbstractString} = Vector{AbstractString}(0))::Dataset
	if(!issorted(urlIDs))
		permutation = sortperm(urlIDs);
		features = features[:, permutation];
		labels = labels[permutation];
		urlIDs = urlIDs[permutation];
		urlParts = urlParts[permutation];
		if size(info, 1) != 0;
			info = info[permutation];
		end
	end

	(domainFeatures, pathFeatures, queryFeatures) = map(i->features[:, urlParts .== i], 1:3);
	(domainBags, pathBags, queryBags) = map(i->EduNets.findranges(urlIDs[urlParts .== i]), 1:3);

	subbags = EduNets.findranges(urlIDs);
	bagLabels = map(b->maximum(labels[b]), subbags);
	if size(info, 1) != 0;
		bagInfo = map(b->info[b][1], subbags);
	else
		bagInfo = Vector{AbstractString}(0);
	end

	domains = EduNets.SortedSingleBagDataset(domainFeatures, bagLabels, domainBags);
	paths = EduNets.SortedSingleBagDataset(pathFeatures, bagLabels, pathBags);
	queries = EduNets.SortedSingleBagDataset(queryFeatures, bagLabels, queryBags);
	Dataset(domains, paths, queries, bagLabels, DataFrames.DataFrame(url = bagInfo))
end

function getindex(dataset::Dataset, i::Int)::Dataset
	return getindex(dataset, [i]);
end

function getindex(dataset::Dataset, indices::Vector{Int})::Dataset
	if size(dataset.info, 1) == 0
		info = DataFrames.DataFrame(url = Vector{AbstractString}(0));
	else
		info = dataset.info[indices, :];
	end
	Dataset(dataset.domains[indices], dataset.paths[indices], dataset.queries[indices], dataset.y[indices], info)
end

function vcat(d1::Dataset,d2::Dataset)
	Dataset(vcat(d1.domains,d2.domains), vcat(d1.paths,d2.paths), vcat(d1.queries,d2.queries), vcat(d1.y,d2.y), vcat(d1.info, d2.info))
end

function sample(dataset::Dataset, n::Int64)
	indices = sample(1:length(dataset.y), min(n, length(dataset.y)), replace=false);
	return getindex(dataset, indices);
end

function sample(dataset::Dataset, n::Vector{Int})
  classbagids = map(i->findn(dataset.y .==i ), 1:maximum(dataset.y));
  indices = mapreduce(i->sample(classbagids[i], minimum([length(classbagids[i]), n[i]]); replace=false), append!, 1:min(length(classbagids), length(n)));
  return(getindex(dataset, indices));
end
