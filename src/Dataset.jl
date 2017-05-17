import Base: Operators.getindex, vcat;
import EduNets: AbstractDataset, SortedSingleBagDataset, sample, findranges;
import DataFrames.DataFrame;

export Dataset, getindex, vcat, sample;

type Dataset{T<:AbstractFloat}<:AbstractDataset
	domains::SortedSingleBagDataset{T}
	paths::SortedSingleBagDataset{T}
	queries::SortedSingleBagDataset{T}

	labels::Vector{Int}
	info::DataFrame;
end

function Dataset(features::Matrix, labels::Vector{Int}, urlIDs::Vector{Int}, urlParts::Vector{Int}; info::Vector{AbstractString} = Vector{AbstractString}(0), T::DataType = Float32)::Dataset
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
	(domainBags, pathBags, queryBags) = map(i->findranges(urlIDs[urlParts .== i]), 1:3);

	subbags = findranges(urlIDs);
	bagLabels = map(b->maximum(labels[b]), subbags);
	if size(info, 1) != 0;
		bagInfo = map(b->info[b][1], subbags);
	else
		bagInfo = Vector{AbstractString}(0);
	end

	domains = SortedSingleBagDataset(domainFeatures, bagLabels, domainBags);
	paths = SortedSingleBagDataset(pathFeatures, bagLabels, pathBags);
	queries = SortedSingleBagDataset(queryFeatures, bagLabels, queryBags);
	Dataset(domains, paths, queries, bagLabels, DataFrame(url = bagInfo))
end

#=
function featureSize(dataset::Dataset)::Int
	size(dataset.domains.x, 1)
end

function getindex(dataset::Dataset, i::Int)::Dataset
	getindex(dataset, [i])
end
=#

function getindex(dataset::Dataset, indices::Vector{Int})::Dataset
	if size(dataset.info, 1) == 0
		info = DataFrame(url = Vector{AbstractString}(0));
	else
		info = dataset.info[indices, :];
	end
	Dataset(dataset.domains[indices], dataset.paths[indices], dataset.queries[indices], dataset.labels[indices], info)
end

function vcat(d1::Dataset,d2::Dataset)
	Dataset(vcat(d1.domains,d2.domains), vcat(d1.paths,d2.paths), vcat(d1.queries,d2.queries), vcat(d1.labels,d2.labels), vcat(d1.info, d2.info))
end

#=
function sample(dataset::Dataset,n::Int64)
  indexes=sample(1:length(dataset.labels),min(n,length(dataset.labels)),replace=false);
  return(getindex(dataset,indexes));
end
=#

function sample(dataset::Dataset, n::Vector{Int})
  classbagids = map(i->findn(dataset.labels .==i ), 1:maximum(dataset.labels));
  indexes = mapreduce(i->sample(classbagids[i], minimum([length(classbagids[i]), n[i]]); replace=false), append!, 1:min(length(classbagids), length(n)));
  return(getindex(dataset, indexes));
end
