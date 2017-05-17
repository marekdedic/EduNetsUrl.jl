"HEX decoder"
function decodeHEX(input::AbstractString)::AbstractString
	raw = input[5:end];
	splitted = split(raw, ":");
	raw = splitted[1];
	splitted = splitted[2:end];
	outputVec = Vector{Char}(cld(length(raw), 2));
	for i in 1:length(outputVec)
		outputVec[i] = Char(parse(Int, raw[(2i - 1):2i], 16));
	end
	output = AbstractString(outputVec);
	for i in splitted
		output *= i;
	end
	return output;
end

"Separates a given URL into 3 parts - domain, query, and path."
function separateUrl(url::AbstractString)::Tuple{Vector{AbstractString}, Vector{AbstractString}, Vector{AbstractString}}
	if contains(url, "://")
		url = split(url, "://")[2];
	end
	splitted = split(url, "/");
	rawDomain = splitted[1];
	if(startswith(rawDomain, "HEX"))
		domain = Vector{AbstractString}();
		push!(domain, decodeHEX(rawDomain));
	else
		domain = split(rawDomain, ".");
	end
	splitted = splitted[2:end];
	path = Vector{AbstractString}();
	query = Vector{AbstractString}();
	if length(splitted) != 0
		splitted2 = split(splitted[end], "?")
		splitted[end] = splitted2[1];
		if length(splitted2) > 1
			query = split(splitted2[2], "&");
		end
		path = splitted;
	end
	# Optional: add empty string when some part is empty array
	if(length(domain) == 0)
		push!(domain, "");
	end
	if(length(path) == 0)
		push!(path, "");
	end
	if(length(query) == 0)
		push!(query, "");
	end
	return (domain, path, query);
end

function processDataset(urls::Vector{AbstractString}, labels::Vector{Int}; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)::Dataset
	features = Vector{Vector{T}}(0);
	processedLabels = Vector{Int}(0);
	bags = Vector{Int}(0);
	urlParts = Vector{Int}(0);
	info = Vector{AbstractString}(0);

	for j in 1:size(labels, 1)
		(domain, path, query) = separateUrl(urls[j]);
		for i in domain
			push!(features, featureGenerator(i, featureCount; T = T));
			push!(processedLabels, labels[j]);
			push!(bags, j);
			push!(urlParts, 1);
			push!(info, urls[j]);
		end
		for i in path
			push!(features, featureGenerator(i, featureCount; T = T));
			push!(processedLabels, labels[j]);
			push!(bags, j);
			push!(urlParts, 2);
			push!(info, urls[j]);
		end
		for i in query
			push!(features, featureGenerator(i, featureCount; T = T));
			push!(processedLabels, labels[j]);
			push!(bags, j);
			push!(urlParts, 3);
			push!(info, urls[j]);
		end
	end
	return Dataset(hcat(features...), processedLabels, bags, urlParts; info = info);
end
