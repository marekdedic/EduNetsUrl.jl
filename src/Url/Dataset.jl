import Base: Operators.getindex, vcat;
import EduNets: AbstractDataset, SortedSingleBagDataset, sample, findranges;
import StatsBase.sample;
import DataFrames.DataFrame;

export Dataset, getindex, vcat, sample;

type Dataset{T<:AbstractFloat}<:AbstractDataset
	domains::SortedSingleBagDataset{T}
	paths::SortedSingleBagDataset{T}
	queries::SortedSingleBagDataset{T}

	labels::AbstractVector{Int}
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
	subbags = findranges(urlIDs);

	domainFeatures = Vector{Vector{T}}(0);
	pathFeatures = Vector{Vector{T}}(0);
	queryFeatures = Vector{Vector{T}}(0);
	bagLabels = Vector{Int}(length(subbags));
	if size(info, 1) != 0;
		bagInfo = Vector{AbstractString}(length(subbags));
	else
		bagInfo = Vector{AbstractString}(0);
	end
	# TODO: Implement bags
	bags = Vector{UnitRange{Int}}(length(subbags));

	for (i, r) in enumerate(subbags)
		for (j, part) in enumerate(urlParts[r])
			if part == 1
				push!(domainFeatures, features[:, first(r) + j - 1]);
			elseif part == 2
				push!(pathFeatures, features[:, first(r) + j - 1]);
			elseif part == 3
				push!(queryFeatures, features[:, first(r) + j - 1]);
			end
		end
		bagLabels[i] = maximum(labels[r]);
		if size(info, 1) != 0;
			bagInfo[i] = info[r][1];
		end
		bags[i] = i:i;
	end

	domains = SortedSingleBagDataset(hcat(domainFeatures...), bagLabels, bags);
	paths = SortedSingleBagDataset(hcat(pathFeatures...), bagLabels, bags);
	queries = SortedSingleBagDataset(hcat(queryFeatures...), bagLabels, bags);
	Dataset(domains, paths, queries, bagLabels, convert(DataFrame, reshape(bagInfo, length(bagInfo), 1)))
end

#=
function featureSize(dataset::Dataset)::Int
	size(dataset.domains.x, 1)
end

function getindex(dataset::Dataset, i::Int)::Dataset
	getindex(dataset, [i])
end
=#

function getindex(dataset::Dataset, indices::AbstractArray{Int})::Dataset
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
function sample(ds::Dataset,n::Int64)
  indexes=sample(1:length(ds.labels),min(n,length(ds.labels)),replace=false);
  return(getindex(ds,indexes));
end
=#

function sample(ds::Dataset,n::Array{Int64})
  classbagids=map(i->findn(ds.labels.==i),1:maximum(ds.labels));
  indexes=mapreduce(i->sample(classbagids[i],minimum([length(classbagids[i]),n[i]]);replace=false),append!,1:min(length(classbagids),length(n)));
  return(getindex(ds,indexes));
end
